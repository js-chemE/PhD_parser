import os

import numpy as np
import pandas as pd
import pytest

from phd_parser.physisorption.tristar import _extract_bet, _extract_isotherm, _parse_value, read_export

MOCK_FILE_PATH = os.path.join(os.path.dirname(__file__), "tristar", "2026-N-198.XLS")


# ----------------------------------------------------------------
# _parse_value
# ----------------------------------------------------------------

def test_parse_value_with_error_and_unit():
    assert _parse_value("105.1138 \xb1 0.4825 m\xb2/g") == {
        "value": 105.1138,
        "error": 0.4825,
        "unit": "m\xb2/g",
    }


def test_parse_value_with_unit_only():
    assert _parse_value("0.0944 g") == {"value": 0.0944, "unit": "g"}


def test_parse_value_plain_number():
    assert _parse_value("0.9997172") == 0.9997172


def test_parse_value_yes_no():
    assert _parse_value("Yes") is True
    assert _parse_value("No") is False


def test_parse_value_not_reported_marker():
    assert _parse_value("*") is None


def test_parse_value_range():
    assert _parse_value("1.7000 nm to 300.0000 nm") == {"low": 1.7, "high": 300.0, "unit": "nm"}


def test_parse_value_date():
    assert _parse_value("17-06-26 10:45:06") == pd.Timestamp("2026-06-17 10:45:06")


def test_parse_value_free_text_passthrough():
    assert _parse_value("Harkins and Jura") == "Harkins and Jura"


def test_parse_value_none_and_nan():
    assert _parse_value(None) is None
    assert _parse_value(float("nan")) is None


# ----------------------------------------------------------------
# read_export — structure
# ----------------------------------------------------------------

def test_read_export_data_keys():
    result = read_export(MOCK_FILE_PATH)
    data = result["data"]
    assert set(data.keys()) == {"adsorption", "desorption"}
    for branch in data.values():
        assert set(branch.keys()) == {"relative_pressure", "quantity_adsorbed"}
        for arr in branch.values():
            assert isinstance(arr, np.ndarray)
        assert branch["relative_pressure"].shape == branch["quantity_adsorbed"].shape

    adsorption = data["adsorption"]["relative_pressure"]
    # ascending, within [0, 1]
    assert np.all(np.diff(adsorption) > 0)
    assert np.all((adsorption >= 0) & (adsorption <= 1))


def test_read_export_meta_top_level_keys():
    result = read_export(MOCK_FILE_PATH)
    meta = result["meta"]
    assert set(meta.keys()) == {"header", "summary", "analyses", "sample_log"}


def test_read_export_header_scalars():
    result = read_export(MOCK_FILE_PATH)
    scalars = result["meta"]["header"]["scalars"]
    assert scalars["Sample"] == "SK1003"
    assert scalars["Operator"] == "Ivan"
    assert scalars["Sample mass"] == {"value": 0.0944, "unit": "g"}


def test_read_export_summary_report():
    result = read_export(MOCK_FILE_PATH)
    summary = result["meta"]["summary"]
    assert summary is not None
    assert summary["table"] is None
    assert any("summary" in n.lower() for n in summary["notes"])


def test_read_export_analyses_groups_present():
    result = read_export(MOCK_FILE_PATH)
    analyses = result["meta"]["analyses"]
    assert set(analyses.keys()) == {
        "isotherm",
        "bet",
        "t_plot",
        "bjh_adsorption",
        "bjh_desorption",
    }
    for name, entry in analyses.items():
        assert "report" in entry and "plots" in entry
        assert entry["report"] is not None


def test_read_export_bet_plots_present():
    result = read_export(MOCK_FILE_PATH)
    assert "surface_area" in result["meta"]["analyses"]["bet"]["plots"]


def test_read_export_bjh_plots_present():
    bjh_ads = read_export(MOCK_FILE_PATH)["meta"]["analyses"]["bjh_adsorption"]
    assert set(bjh_ads["plots"].keys()) == {"cumulative_pore_volume", "dvdlogw_pore_volume"}


def test_read_export_sample_log():
    sample_log = read_export(MOCK_FILE_PATH)["meta"]["sample_log"]
    table = sample_log["table"]
    assert table["columns"][:3] == ["Date", "Time", "Log Message"]
    assert len(table["rows"]) > 0
    for row in table["rows"]:
        assert isinstance(row[0], str)  # Date
        assert isinstance(row[1], str)  # Time
        assert isinstance(row[2], str)  # Log Message


def test_read_export_rejects_wrong_extension(tmp_path):
    bad_file = tmp_path / "not_a_report.txt"
    bad_file.write_text("hello")
    with pytest.raises(ValueError):
        read_export(bad_file)


# ----------------------------------------------------------------
# read_export — BET
# ----------------------------------------------------------------

def test_read_export_bet_values():
    bet = read_export(MOCK_FILE_PATH)["bet"]
    assert bet["surface_area"] == pytest.approx(105.1138)
    assert bet["surface_area_error"] == pytest.approx(0.4825)
    assert bet["correlation_coefficient"] == pytest.approx(0.9997172)
    assert bet["monolayer_capacity"] == pytest.approx(24.1498)


def test_extract_bet_returns_none_for_missing_section():
    assert _extract_bet(None) is None


# ----------------------------------------------------------------
# _extract_isotherm
# ----------------------------------------------------------------

def test_extract_isotherm_returns_empty_dict_for_missing_section():
    assert _extract_isotherm(None) == {}


def test_extract_isotherm_splits_at_turnover():
    section = read_export(MOCK_FILE_PATH)["meta"]["analyses"]["isotherm"]["report"]
    out = _extract_isotherm(section)
    # the adsorption branch includes the turnover point, so its maximum
    # relative pressure must be at least as high as the desorption branch's
    assert out["adsorption"]["relative_pressure"].max() >= out["desorption"]["relative_pressure"].max()
    assert out["adsorption"]["quantity_adsorbed"].min() > 0
