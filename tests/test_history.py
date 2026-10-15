"""The processing history: recorded per step, never on the input, and kept through to_netcdf / from_netcdf.

Also checks that the older tracking of each class (IR merge_log and baseline
parameters, MS trace_corrections, LV metadata, GC provenance) survives storing.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from phd_parser import GCData, IRData, LVData, MSData, RamanData
from phd_parser.history import decode_attr, encode_attr, format_history, suppress_recording

AXIS_PER_CM = np.linspace(1000.0, 3000.0, 201)
MS_FILE = os.path.join(os.path.dirname(__file__), "massspec", "2026-03-26_sk1002_02-react.asc")


def _steps(obj) -> list[str]:
    return [entry["step"] for entry in obj.history]


@pytest.fixture
def ir():
    rng = np.random.default_rng(0)
    values = 1.0 + 0.01 * rng.standard_normal((6, AXIS_PER_CM.size))
    return IRData.from_arrays(AXIS_PER_CM, values, tos=np.arange(6) * 60.0, tos_start="2025-05-15 10:00")


def test_each_step_is_recorded_once_with_its_parameters(ir):
    processed = ir.smooth_savgol(11, 2).select_wavenumber_range(1200, 2800)
    assert _steps(processed) == ["from_arrays", "smooth_savgol", "select_wavenumber_range"]
    assert processed.history[1]["parameters"] == {"window_length": 11, "polyorder": 2}
    assert processed.history[2]["parameters"] == {"min_cm": 1200, "max_cm": 2800}
    assert {"time", "version"} <= set(processed.history[0])


def test_the_input_is_never_touched(ir):
    ir.smooth_moving(3)
    ir.select_tos_range()  # returns the same Dataset object internally
    assert _steps(ir) == ["from_arrays"]


def test_only_the_outermost_call_is_recorded(ir):
    corrected = ir.correct_baseline((2500, 2600), control_points_cm=[1100, 2000, 2900])
    assert _steps(corrected) == ["from_arrays", "correct_baseline"]
    with suppress_recording():
        quiet = ir.smooth_moving(3)
    assert _steps(quiet) == ["from_arrays"]


def test_a_no_op_records_nothing(ir):
    assert ir.to_single_beam() is ir


def test_merge_records_the_other_segments_history(ir):
    later = ir.move_tos_start_by(-3600).smooth_moving(3)
    merged = ir.merge(later)
    entry = merged.history[-1]
    assert entry["step"] == "merge"
    assert [step["step"] for step in entry["parameters"]["other"]["history"]] == [
        "from_arrays", "move_tos_start_by", "smooth_moving"
    ]


def test_ir_history_and_tracking_survive_netcdf(ir, tmp_path):
    later = ir.move_tos_start_by(-3600)
    processed = (
        ir.merge(later)
        .correct_offset((2500, 2600))
        .average_scans_by_tos([60.0, 120.0], number_of_scans=2)  # writes a None attr
    )
    processed.to_netcdf(tmp_path / "ir.nc")
    back = IRData.from_netcdf(tmp_path / "ir.nc")

    assert back.history[:-1] == processed.history
    assert back.history[-1]["step"] == "from_netcdf"
    assert json.loads(back.ds.attrs["merge_log"]) == json.loads(processed.ds.attrs["merge_log"])
    assert list(back.ds.attrs["baseline_anchor_range_cm"]) == [2500, 2600]
    assert back.ds.attrs["averaged_time_window"] is None
    np.testing.assert_allclose(back.values, processed.values)
    np.testing.assert_allclose(back.baseline, processed.baseline)


def test_raman_history_and_metadata_survive_netcdf(tmp_path):
    raman = RamanData.from_arrays(
        AXIS_PER_CM, np.ones((3, AXIS_PER_CM.size)), excitation_wavelength_nm=532.0,
        tos=np.arange(3.0), metadata={"instrument": "BTC655N", "laser_power": [10, "mW"]},
    ).smooth_moving(3)
    raman.to_netcdf(tmp_path / "raman.nc")
    back = RamanData.from_netcdf(tmp_path / "raman.nc")
    assert _steps(back) == ["from_arrays", "smooth_moving", "from_netcdf"]
    assert back.metadata == {"instrument": "BTC655N", "laser_power": [10, "mW"]}
    assert back.excitation_wavelength_nm == 532.0


def test_gc_history_survives_netcdf(tmp_path):
    rt = np.arange(0, 60, 0.05)
    gc = GCData.from_arrays(rt, np.exp(-((rt - 30) ** 2)), detector="fid", unit="pA",
                            injection_time="2025-05-15 16:10:51")
    cropped = gc.select_retention_time_range(10, 50)
    cropped.to_netcdf(tmp_path / "gc.nc")
    back = GCData.from_netcdf(tmp_path / "gc.nc")
    assert _steps(back) == ["from_arrays", "select_retention_time_range", "from_netcdf"]


def test_ms_history_and_trace_corrections_survive_netcdf(tmp_path):
    ms = MSData.from_quadstar_asc(MS_FILE).correct_traces()
    ms.to_netcdf(tmp_path / "ms.nc")
    back = MSData.from_netcdf(tmp_path / "ms.nc")
    assert _steps(back) == ["from_quadstar_asc", "correct_traces", "from_netcdf"]
    assert [entry["method"] for entry in back.ds.attrs["trace_corrections"]] == ["min_shift"]


def test_lv_history_and_metadata_survive_netcdf(tmp_path):
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01 10:00", periods=4, freq="10s"),
        "F1 CH4 PV": [1.0, 2.0, 3.0, 4.0],
    })
    lv = LVData.from_dataframe(frame, channel_meta={"F1 CH4 PV": {"unit": "mL/min", "species": "Methan"}})
    renamed = lv.rename_species().smooth_moving(2)
    assert _steps(renamed) == ["from_dataframe", "rename_species", "smooth_moving"]

    renamed.to_netcdf(tmp_path / "lv.nc")
    back = LVData.from_netcdf(tmp_path / "lv.nc")
    assert _steps(back) == ["from_dataframe", "rename_species", "smooth_moving", "from_netcdf"]
    # lived only in metadata and was lost on storing before
    assert back.metadata["species_convention"] == "phd_standard"
    assert back.ds["F1 CH4 PV"].attrs["species"] == "CH4"


@pytest.mark.parametrize("value", [None, True, {"a": [1, 2]}, ["x", "y"], [], "Methan → CH₄"])
def test_attrs_netcdf_cannot_hold_round_trip_through_json(value):
    encoded = encode_attr(value)
    assert isinstance(encoded, str) and encoded.startswith("json:")
    assert decode_attr(encoded) == value


@pytest.mark.parametrize("value", [1, 2.5, "plain", [1.0, 2.0]])
def test_attrs_netcdf_holds_stay_as_they_are(value):
    assert encode_attr(value) == value


def test_format_history_is_one_line_per_step(ir):
    text = format_history(ir.smooth_savgol(11, 2).history)
    lines = text.splitlines()
    assert len(lines) == 2
    assert lines[1].endswith("smooth_savgol(window_length=11, polyorder=2)")
    assert ir.format_history().startswith("1. ")
