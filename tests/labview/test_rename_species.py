"""LVData.rename_species renames the per-channel species attribute only."""
import pandas as pd

from phd_parser.labview import LVData


def test_rename_species_changes_only_the_attribute():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01 10:00", periods=3, freq="10s"),
        "F1 CH4 PV": [1.0, 2.0, 3.0],
        "T oven": [200.0, 200.0, 200.0],
    })
    lv = LVData.from_dataframe(frame, channel_meta={"F1 CH4 PV": {"unit": "mL/min", "species": "Methan"}})
    renamed = lv.rename_species()
    assert renamed.ds["F1 CH4 PV"].attrs["species"] == "CH4"
    assert lv.ds["F1 CH4 PV"].attrs["species"] == "Methan"
    assert "species" not in renamed.ds["T oven"].attrs
    assert renamed.metadata["species_convention"] == "phd_standard"
    assert renamed.ds["F1 CH4 PV"].values.tolist() == [1.0, 2.0, 3.0]
