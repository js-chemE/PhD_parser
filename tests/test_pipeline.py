"""Pipelines: a list of steps that runs on any data class, round-trips through JSON, and is read off a history."""
from pathlib import Path

import numpy as np
import pytest

import phd_parser as pp
from phd_parser import Pipeline, Step
from phd_parser.spectroscopy import IRData, SpectroscopyData, XPSData

AXIS_PER_CM = np.linspace(1000.0, 3000.0, 201)
VMS = Path(__file__).parent / "spectroscopy" / "xps" / "vamas" / "SK-0001_3.vms"


@pytest.fixture
def series():
    peak = np.exp(-((AXIS_PER_CM - 2000.0) / 50.0) ** 2)
    values = np.outer(np.arange(1.0, 7.0), peak) + 0.1
    return IRData.from_arrays(AXIS_PER_CM, values, tos=np.arange(6) * 10.0, data_type="absorbance")


def test_steps_read_every_spelling():
    assert Step.coerce("normalise_max") == Step(method="normalise_max")
    assert Step.coerce(("smooth_savgol", {"window_length": 5})) == Step(method="smooth_savgol", parameters={"window_length": 5})
    assert Step.coerce({"smooth_savgol": {"window_length": 5}}).parameters == {"window_length": 5}
    assert Step.coerce({"method": "sort", "parameters": None}).parameters == {}
    with pytest.raises(ValueError):
        Step.coerce(("a", "b", "c"))


def test_pipeline_equals_the_chained_calls(series):
    pipeline = Pipeline([
        ("select_tos_range", {"min_s": 0, "max_s": 30}),
        ("smooth_savgol", {"window_length": 5, "polyorder": 2}),
        "normalise_max",
    ], name="test")
    result = pipeline(series)
    chained = series.select_tos_range(0, 30).smooth_savgol(window_length=5, polyorder=2).normalise_max()
    np.testing.assert_allclose(result.values, chained.values)
    assert [entry["step"] for entry in result.history] == ["from_arrays", "select_tos_range", "smooth_savgol", "normalise_max"]
    assert len(pipeline) == 3 and series.shape[0] == 6  # the input is untouched
    assert "smooth_savgol(window_length=5, polyorder=2)" in repr(pipeline)


def test_pipeline_is_immutable_and_composable(series):
    base = Pipeline(["normalise_max"], name="base")
    longer = base.append(("smooth_moving", {"window_size": 3}))
    assert len(base) == 1 and len(longer) == 2
    both = base + Pipeline([("select_wavenumber_range", {"min_cm": 1500, "max_cm": 2500})])
    assert both.name == "base" and both[1].method == "select_wavenumber_range"
    assert isinstance(both[:1], Pipeline) and len(both[:1]) == 1
    np.testing.assert_allclose(both(series).values, series.normalise_max().select_wavenumber_range(1500, 2500).values)


def test_pipeline_runs_on_every_class_and_maps_over_dicts(series):
    regions = XPSData.from_vamas(VMS)
    xps_pipeline = Pipeline(["to_cps", ("select_binding_energy_range", {"min_eV": 280, "max_eV": 292})])
    processed = xps_pipeline.map({name: region for name, region in regions.items() if name == "C1s Scan"})
    assert set(processed) == {"C1s Scan"} and processed["C1s Scan"].data_type == "cps"
    as_list = Pipeline(["normalise_max"]).map([series, series.select_by_idx(0)])
    assert len(as_list) == 2 and as_list[1].ndim == 1
    with pytest.raises(AttributeError, match="XPSData has no method 'to_absorbance'"):
        Pipeline(["to_absorbance"])(regions["C1s Scan"])


def test_pipeline_from_history_replays_the_processing(series):
    processed = series.select_tos_range(10, 40).correct_offset((2800, 3000)).normalise_value(2.0)
    pipeline = Pipeline.from_history(processed, name="replayed")
    assert [step.method for step in pipeline] == ["select_tos_range", "correct_offset", "normalise_value"]
    assert pipeline[0].parameters == {"min_s": 10, "max_s": 40, "empty": "warn"}
    again = pipeline(series)
    np.testing.assert_allclose(again.values, processed.values)
    # the constructor's axis array is kept in the history only as a summary, so it cannot be replayed
    with pytest.raises(ValueError, match="from_arrays"):
        Pipeline.from_history(processed, include_constructor=True)
    short = IRData.from_arrays(np.linspace(1000.0, 1100.0, 5), np.ones(5)).normalise_max()
    assert [step.method for step in Pipeline.from_history(short, include_constructor=True)] == ["from_arrays", "normalise_max"]


def test_pipeline_from_history_refuses_summarised_arguments(series):
    background = np.linspace(1.0, 2.0, AXIS_PER_CM.size)  # longer than 20 values: stored as a summary
    processed = SpectroscopyData(ds=series.ds.assign_attrs(data_type="single_beam")).with_background(background)
    with pytest.raises(ValueError, match="with_background.*background.*summary"):
        Pipeline.from_history(processed)


def test_pipeline_json_round_trip(tmp_path, series):
    pipeline = Pipeline([
        ("select_tos_range", {"min_s": 0, "max_s": 30}),
        ("correct_baseline", {"anchor_range_cm": (2800, 3000), "control_points_cm": [1100, 1800, 2900]}),
    ], name="drifts")
    path = tmp_path / "pipeline.json"
    text = pipeline.to_json(path)
    assert path.exists() and '"name": "drifts"' in text
    back = Pipeline.from_json(path)
    assert back.name == "drifts" and [step.method for step in back] == [step.method for step in pipeline]
    assert back[1].parameters["control_points_cm"] == [1100, 1800, 2900]
    np.testing.assert_allclose(back(series).values, pipeline(series).values)
    assert Pipeline.from_json(text) == back
    assert pp.Pipeline is Pipeline


def test_from_history_since_the_last_load_and_nested_lists(tmp_path, series):
    processed = series.mask_wavenumber_ranges([(2220, 2420)]).normalise_max()
    processed.to_netcdf(tmp_path / "stage.nc")
    loaded = SpectroscopyData.from_netcdf(tmp_path / "stage.nc").smooth_moving(3).select_wavenumber_range(1500, 2500)
    stage = Pipeline.from_history(loaded, since="from_netcdf")
    assert [step.method for step in stage] == ["smooth_moving", "select_wavenumber_range"]
    whole = Pipeline.from_history(loaded)
    assert whole[0].method == "mask_wavenumber_ranges" and whole[0].parameters == {"ranges_cm": [[2220, 2420]]}
    np.testing.assert_allclose(whole(series).values, loaded.values, equal_nan=True)
    with pytest.raises(ValueError, match="No step 'merge'"):
        Pipeline.from_history(loaded, since="merge")
    # a processed stage saved and loaded again: its own load at the end does not hide its steps
    loaded.to_netcdf(tmp_path / "stage2.nc")
    reloaded = SpectroscopyData.from_netcdf(tmp_path / "stage2.nc")
    assert [step.method for step in Pipeline.from_history(reloaded, since="from_netcdf")] == ["smooth_moving", "select_wavenumber_range"]
    first_stage = SpectroscopyData.from_netcdf(tmp_path / "stage.nc")
    assert [step.method for step in Pipeline.from_history(first_stage, since="from_netcdf")] == ["mask_wavenumber_ranges", "normalise_max"]
