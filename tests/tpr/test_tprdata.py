"""TPRData: both column variants of the LabVIEW TPR rig's export, processing, history and NetCDF."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phd_parser.tpr import TPRData, labview_tcd

FIXTURES = Path(__file__).parent / "labview_tcd"
VARIANT_1 = FIXTURES / "tpr3010_v1.txt"                      # time (s), TCD (H2), FID, TCD (O2), TC 1
VARIANT_1_UNNAMED = FIXTURES / "tpr3088_v1_unnamed-columns.txt"  # the same plus seven columns without a name
VARIANT_2 = FIXTURES / "tpx3098_v2.txt"                      # date, time, time (s), TCD (H2), TCD (O2), TC 1


def _steps(obj) -> list[str]:
    return [entry["step"] for entry in obj.history]


# ----------------------------------------------------------------
# The parser
# ----------------------------------------------------------------

def test_read_export_variant_1_with_fid():
    raw = labview_tcd.read_export(VARIANT_1)
    data, meta = raw["data"], raw["meta"]
    assert meta["format_variant"] == 1
    assert meta["username"] == "Siddhant" and meta["sample_name"] == "MoO3(20)/TiO2"
    assert meta["sample_weight_mg"] == pytest.approx(104.0)
    assert meta["comment"].startswith("TPR 300C 5C/min")
    assert meta["start_time"] == pd.Timestamp("2026-04-10 10:30:23") and meta["start_time"].tz is None
    assert meta["columns"] == ["time (s)", "TCD (H2)", "FID", "TCD (O2)", "TC 1"]
    assert meta["column_map"] == {"time (s)": "time_s", "TCD (H2)": "tcd_h2", "FID": "fid",
                                  "TCD (O2)": "tcd_o2", "TC 1": "temperature_C"}
    assert set(data["signals"]) == {"tcd_h2", "tcd_o2", "fid"} and data["extra"] == {}
    assert "timestamp" not in data
    assert data["time_s"][0] == pytest.approx(0.3) and data["temperature_C"][0] == pytest.approx(26.7)
    assert data["signals"]["tcd_h2"][0] == pytest.approx(-3.01) and data["signals"]["tcd_o2"][0] == pytest.approx(0.95)
    assert data["time_s"].size == meta["n_rows"] == 502


def test_read_export_variant_2_with_timestamps_and_decimal_comma():
    raw = labview_tcd.read_export(VARIANT_2)
    data, meta = raw["data"], raw["meta"]
    assert meta["format_variant"] == 2
    assert meta["sample_name"] == "Pd(0.25)/MoO3(20)/TiO2"
    assert meta["sample_weight_mg"] == pytest.approx(101.9)  # '"101,9 mg"'
    assert meta["start_time"] == pd.Timestamp("2026-07-23 11:33:32")
    assert meta["columns"] == ["date", "time", "time (s)", "TCD (H2)", "TCD (O2)", "TC 1"]
    assert set(data["signals"]) == {"tcd_h2", "tcd_o2"}  # no FID any more
    assert data["timestamp"].dtype == np.dtype("datetime64[ns]")
    assert data["timestamp"][0] == np.datetime64("2026-07-23T11:33:32")
    # the per-row clock and the elapsed seconds agree to the second
    elapsed = (data["timestamp"] - data["timestamp"][0]) / np.timedelta64(1, "s")
    np.testing.assert_allclose(elapsed, data["time_s"] - data["time_s"][0], atol=1.5)


def test_sample_weight_parsing():
    assert labview_tcd.parse_sample_weight_mg("104.0 mg") == pytest.approx(104.0)
    assert labview_tcd.parse_sample_weight_mg('"101,9 mg"') == pytest.approx(101.9)
    assert labview_tcd.parse_sample_weight_mg("0.1 g") == pytest.approx(100.0)
    assert labview_tcd.parse_sample_weight_mg("") is None and labview_tcd.parse_sample_weight_mg(None) is None


def test_unnamed_extra_columns_warn_but_do_not_fail(caplog):
    with caplog.at_level(logging.WARNING, logger="phd_parser.tpr.labview_tcd"):
        raw = labview_tcd.read_export(VARIANT_1_UNNAMED)
    assert raw["meta"]["format_variant"] == 1 and set(raw["data"]["signals"]) == {"tcd_h2", "tcd_o2", "fid"}
    assert list(raw["data"]["extra"]) == [f"column_{n}" for n in range(6, 13)]
    assert raw["meta"]["extra_columns"]["column_6"] == 6
    assert raw["data"]["extra"]["column_6"][0] == pytest.approx(4.83e-13)
    assert any("without a name" in record.message and "tpr3088" in record.message for record in caplog.records)
    run = TPRData.from_labview_tcd(VARIANT_1_UNNAMED)
    assert "column_6" in run.signals and run.get_signal("column_6").attrs["units"] == "unknown"
    assert run.sample_weight_mg == pytest.approx(102.1)


def test_unknown_named_column_is_kept_with_a_warning(tmp_path, caplog):
    text = VARIANT_1.read_text(encoding="utf-8")
    lines = text.splitlines()
    column_index = next(index for index, line in enumerate(lines) if line.startswith("time (s)"))
    lines[column_index] = "time (s)\tTCD (H2)\tFID\tTCD (O2)\tTC 1\tMFC H2 (ml/min)"
    rows = lines[column_index + 2:column_index + 12]
    body = [line.rstrip("\t") + "\t25.0" for line in rows]
    path = tmp_path / "tpr_with_mfc.txt"
    path.write_text("\n".join(lines[:column_index + 2] + body) + "\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="phd_parser.tpr.labview_tcd"):
        raw = labview_tcd.read_export(path)
    assert raw["meta"]["column_map"]["MFC H2 (ml/min)"] == "mfc_h2_ml_min"
    assert raw["meta"]["extra_columns"] == {"mfc_h2_ml_min": "MFC H2 (ml/min)"}
    np.testing.assert_allclose(raw["data"]["extra"]["mfc_h2_ml_min"], 25.0)
    assert any("MFC H2 (ml/min)" in record.message for record in caplog.records)
    run = TPRData.from_labview_tcd(path)
    assert set(run.signals) == {"tcd_h2", "tcd_o2", "fid", "mfc_h2_ml_min"}
    assert run.get_signal("mfc_h2_ml_min").attrs["source_column"] == "MFC H2 (ml/min)"


def test_a_file_without_the_column_line_is_refused(tmp_path):
    path = tmp_path / "not_tpr.txt"
    path.write_text("username\tx\n1\t2\t3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="time \\(s\\)"):
        labview_tcd.read_export(path)


# ----------------------------------------------------------------
# The data class
# ----------------------------------------------------------------

def test_from_labview_tcd_builds_the_dataset_for_both_variants():
    first = TPRData.from_labview_tcd(VARIANT_1)
    assert first.signals == ["tcd_h2", "tcd_o2", "fid"]
    assert first.sample_name == "MoO3(20)/TiO2" and first.sample_weight_mg == pytest.approx(104.0)
    assert first.start_time == pd.Timestamp("2026-04-10 10:30:23")
    assert first.ds.attrs["technique"] == "tpr" and first.ds.attrs["format_variant"] == 1
    assert first.ds.attrs["columns"] == ["time (s)", "TCD (H2)", "FID", "TCD (O2)", "TC 1"]
    assert first.ds[labview_tcd.COLUMN_MAP["TCD (H2)"]].attrs == {"units": "mV", "source_column": "TCD (H2)"}
    assert first.ds["temperature"].attrs["units"] == "°C" and first.ds["temperature"].attrs["source_column"] == "TC 1"
    assert first.timestamps is None
    assert _steps(first) == ["from_labview_tcd"]
    assert first.history[0]["parameters"]["filepath"].endswith("tpr3010_v1.txt")

    second = TPRData.from_labview_tcd(VARIANT_2)
    assert second.signals == ["tcd_h2", "tcd_o2"]
    assert second.sample_weight_mg == pytest.approx(101.9)
    assert second.start_time == pd.Timestamp("2026-07-23 11:33:32")
    assert second.timestamps is not None and second.timestamps[0] == np.datetime64("2026-07-23T11:33:32")
    with pytest.raises(KeyError, match="tcd_h2"):
        second.get_signal("fid")
    assert "TPRData(sample_name='Pd(0.25)/MoO3(20)/TiO2'" in repr(second) and "tcd_h2" in repr(second)


def test_select_time_range_is_immutable_and_recorded():
    run = TPRData.from_labview_tcd(VARIANT_2)
    window = run.select_time_range(1000, 2000)
    assert window.time.min() >= 1000 and window.time.max() <= 2000 and 0 < window.n_points < run.n_points
    assert _steps(window) == ["from_labview_tcd", "select_time_range"]
    assert window.history[-1]["parameters"] == {"min_s": 1000, "max_s": 2000, "empty": "warn"}
    assert _steps(run) == ["from_labview_tcd"] and run.time.min() < 1000
    assert window.timestamps is not None and window.timestamps.size == window.n_points
    assert run.select_time_range(max_s=100).n_points == int((run.time <= 100).sum())
    empty = run.select_time_range(1e6, 2e6)
    assert empty.n_points == 0
    with pytest.raises(ValueError):
        run.select_time_range(1e6, 2e6, empty="raise")


def test_select_temperature_range_keeps_both_passes_in_time_order():
    run = TPRData.from_labview_tcd(VARIANT_2)
    window = run.select_temperature_range(300, 400)
    assert window.temperature.min() >= 300 and window.temperature.max() <= 400
    assert np.all(np.diff(window.time) > 0)
    # heating and cooling both pass 300-400 °C, so there is a gap in time
    assert np.diff(window.time).max() > 10 * np.median(np.diff(window.time))
    assert _steps(window)[-1] == "select_temperature_range"


def _synthetic_run(slope: float = 0.001, offset: float = 0.5, area: float = 60.0, centre: float = 500.0,
                   sigma: float = 40.0, n: int = 1001) -> tuple[TPRData, np.ndarray]:
    time = np.linspace(0.0, 1000.0, n)
    peak = area / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((time - centre) / sigma) ** 2)
    signal = offset + slope * time + peak
    temperature = 30.0 + 10.0 / 60.0 * time
    run = TPRData.from_arrays(time, temperature, {"tcd_h2": signal, "tcd_o2": np.full(n, 0.55)},
                              sample_name="synthetic", sample_weight_mg=100.0, start_time="2026-01-01 08:00")
    return run, peak


def test_subtract_baseline_linear_and_constant():
    run, peak = _synthetic_run()
    linear = run.subtract_baseline("tcd_h2", "linear", anchor_points_s=[100, 900])
    assert "tcd_h2_baseline" in linear.ds and linear.signals == ["tcd_h2", "tcd_o2"]
    np.testing.assert_allclose(linear.ds["tcd_h2_baseline"].values, 0.5 + 0.001 * run.time, atol=1e-6)
    np.testing.assert_allclose(linear.get_signal("tcd_h2").values, peak, atol=1e-6)
    assert linear.get_signal("tcd_h2").attrs["baseline_method"] == "linear"
    assert linear.ds["tcd_h2_baseline"].attrs["baseline_of"] == "tcd_h2"
    assert _steps(linear) == ["from_arrays", "subtract_baseline"]
    assert linear.history[-1]["parameters"]["anchor_points_s"] == [100, 900]
    assert "tcd_h2_baseline" not in run.ds  # the input is untouched

    fitted = run.subtract_baseline("tcd_h2", "linear", time_window_s=(0, 200))
    np.testing.assert_allclose(fitted.ds["tcd_h2_baseline"].values, 0.5 + 0.001 * run.time, atol=1e-3)

    flat, _ = _synthetic_run(slope=0.0, offset=2.0)
    constant = flat.subtract_baseline("tcd_h2", "constant", time_window_s=(0, 200))
    np.testing.assert_allclose(constant.ds["tcd_h2_baseline"].values, 2.0, atol=1e-6)
    # a second subtraction accumulates into the stored baseline
    again = constant.subtract_baseline("tcd_h2", "constant", anchor_points_s=[950])
    np.testing.assert_allclose(again.ds["tcd_h2_baseline"].values, 2.0, atol=1e-6)
    with pytest.raises(ValueError):
        run.subtract_baseline("tcd_h2", "linear")
    with pytest.raises(ValueError):
        run.subtract_baseline("tcd_h2", "linear", time_window_s=(0, 200), anchor_points_s=[100])
    with pytest.raises(ValueError):
        run.subtract_baseline("tcd_h2", "quadratic", time_window_s=(0, 200))


def test_integrate_recovers_the_area_of_a_synthetic_peak():
    run, peak = _synthetic_run(slope=0.0, offset=0.0, area=60.0)
    assert run.integrate("tcd_h2") == pytest.approx(60.0, rel=1e-4)
    assert run.integrate("tcd_h2", 400, 600) == pytest.approx(60.0 * 0.9876, rel=2e-3)  # +-2.5 sigma
    drifting, _ = _synthetic_run(area=60.0)
    corrected = drifting.subtract_baseline("tcd_h2", "linear", anchor_points_s=[100, 900])
    assert corrected.integrate("tcd_h2") == pytest.approx(60.0, rel=1e-3)
    with pytest.raises(ValueError):
        run.integrate("tcd_h2", 2000, 3000)


def test_scale_and_per_gram_touch_the_signal_and_its_baseline():
    run, _ = _synthetic_run()
    corrected = run.subtract_baseline("tcd_h2", "linear", anchor_points_s=[100, 900])
    flipped = corrected.scale("tcd_h2", -1)
    np.testing.assert_allclose(flipped.get_signal("tcd_h2").values, -corrected.get_signal("tcd_h2").values)
    np.testing.assert_allclose(flipped.ds["tcd_h2_baseline"].values, -corrected.ds["tcd_h2_baseline"].values)
    assert flipped.get_signal("tcd_h2").attrs["scaled_by"] == -1.0
    assert flipped.scale("tcd_h2", -1).get_signal("tcd_h2").attrs["scaled_by"] == 1.0
    np.testing.assert_allclose(flipped.get_signal("tcd_o2").values, corrected.get_signal("tcd_o2").values)
    per_gram = corrected.per_gram("tcd_h2")
    np.testing.assert_allclose(per_gram.get_signal("tcd_h2").values, corrected.get_signal("tcd_h2").values / 0.1)
    assert per_gram.get_signal("tcd_h2").attrs["units"] == "mV g^-1"
    assert per_gram.integrate("tcd_h2") == pytest.approx(10 * corrected.integrate("tcd_h2"))
    with pytest.raises(ValueError):
        per_gram.per_gram("tcd_h2")
    no_mass = TPRData.from_arrays(run.time, run.temperature, {"tcd_h2": run.get_signal("tcd_h2").values})
    with pytest.raises(ValueError):
        no_mass.per_gram("tcd_h2")
    assert _steps(per_gram) == ["from_arrays", "subtract_baseline", "per_gram"]


def test_smooth_moving_skips_nan_and_leaves_the_temperature_alone():
    time = np.arange(20.0)
    signal = np.ones(20)
    signal[10] = np.nan
    run = TPRData.from_arrays(time, 30 + time, {"tcd_h2": signal})
    smoothed = run.smooth_moving(5)
    # the NaN sample is skipped, not spread: its neighbours fill the window
    np.testing.assert_allclose(smoothed.get_signal("tcd_h2").values, 1.0)
    assert np.isnan(run.get_signal("tcd_h2").values).sum() == 1  # the input keeps its gap
    np.testing.assert_allclose(smoothed.temperature, run.temperature)
    assert smoothed.get_signal("tcd_h2").attrs["smooth_moving_window"] == 5
    assert _steps(smoothed) == ["from_arrays", "smooth_moving"]
    with pytest.raises(ValueError):
        run.smooth_moving(0)


def test_heating_rate_of_the_real_ramps():
    # tpx3098: programmed 10 °C min^-1 to 700 °C after a 15 min dwell
    assert TPRData.from_labview_tcd(VARIANT_2).heating_rate() == pytest.approx(10.0, abs=1.0)
    # tpr3010: programmed 5 °C min^-1 to 300 °C, the bed thermocouple follows at about 4.4 K min^-1
    assert TPRData.from_labview_tcd(VARIANT_1).heating_rate() == pytest.approx(4.4, abs=0.7)
    synthetic, _ = _synthetic_run()  # exactly 10 K min^-1 everywhere
    assert synthetic.heating_rate(window_s=10) == pytest.approx(10.0, rel=1e-6)
    isothermal = TPRData.from_arrays(np.arange(100.0), np.full(100, 30.0), {"tcd_h2": np.zeros(100)})
    assert np.isnan(isothermal.heating_rate())


def test_netcdf_round_trip_keeps_values_attrs_and_history(tmp_path):
    run = TPRData.from_labview_tcd(VARIANT_2).select_time_range(1000, 6000)
    processed = run.subtract_baseline("tcd_h2", "linear", anchor_points_s=[1100, 5900]).per_gram("tcd_h2")
    path = tmp_path / "tpx3098.nc"
    processed.to_netcdf(path)
    loaded = TPRData.from_netcdf(path)
    assert _steps(loaded) == ["from_labview_tcd", "select_time_range", "subtract_baseline", "per_gram", "from_netcdf"]
    assert loaded.history[2]["parameters"]["anchor_points_s"] == [1100, 5900]
    np.testing.assert_allclose(loaded.time, processed.time)
    np.testing.assert_allclose(loaded.temperature, processed.temperature)
    for name in ("tcd_h2", "tcd_o2", "tcd_h2_baseline"):
        np.testing.assert_allclose(loaded.ds[name].values, processed.ds[name].values)
        assert loaded.ds[name].attrs == processed.ds[name].attrs
    assert loaded.ds["temperature"].attrs["units"] == "°C"
    assert loaded.sample_name == processed.sample_name and loaded.sample_weight_mg == pytest.approx(101.9)
    assert loaded.start_time == pd.Timestamp("2026-07-23 11:33:32")
    assert loaded.ds.attrs["columns"] == processed.ds.attrs["columns"]
    assert loaded.ds.attrs["format_variant"] == 2
    np.testing.assert_array_equal(loaded.timestamps, processed.timestamps)
    # the file is closed after reading: it can be overwritten at once
    loaded.to_netcdf(path)


def test_to_dataframe():
    run = TPRData.from_labview_tcd(VARIANT_2).subtract_baseline("tcd_h2", "constant", time_window_s=(0, 600))
    frame = run.to_dataframe()
    assert frame.index.name == "time_s" and len(frame) == run.n_points
    assert list(frame.columns) == ["timestamp", "temperature_C", "tcd_h2", "tcd_o2", "tcd_h2_baseline"]
    assert frame["timestamp"].iloc[0] == pd.Timestamp("2026-07-23 11:33:32")
    plain = TPRData.from_labview_tcd(VARIANT_1).to_dataframe()
    assert list(plain.columns) == ["temperature_C", "tcd_h2", "tcd_o2", "fid"]


def test_from_arrays_refuses_mismatched_shapes_and_reserved_names():
    time = np.arange(10.0)
    with pytest.raises(ValueError):
        TPRData.from_arrays(time, np.zeros(9), {"tcd_h2": np.zeros(10)})
    with pytest.raises(ValueError):
        TPRData.from_arrays(time, np.zeros(10), {"tcd_h2": np.zeros(11)})
    with pytest.raises(ValueError):
        TPRData.from_arrays(time, np.zeros(10), {"temperature": np.zeros(10)})
    with pytest.raises(ValueError):
        TPRData.from_arrays(time, np.zeros(10), {"tcd_h2_baseline": np.zeros(10)})
