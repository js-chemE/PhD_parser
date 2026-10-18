"""LVData resampling and smoothing: NaN channels (a column an older file did not record) and the edges."""
import numpy as np
import pandas as pd

from phd_parser.labview import LVData

TOS_START = pd.Timestamp("2026-04-21 14:00:00", tz="Europe/Amsterdam")


def make_lv():
    tos = np.arange(8) * 30.0
    frame = pd.DataFrame({
        "timestamp": TOS_START + pd.to_timedelta(tos, unit="s"),
        "Reactor T PV": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
        "F1 CO PV": [np.nan, np.nan, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0],
    })
    return LVData.from_dataframe(frame, tos_start=TOS_START)


def test_smooth_moving_keeps_the_edges_and_does_not_spread_nan():
    smoothed = make_lv().smooth_moving(3)
    temperature = smoothed.get_channel("Reactor T PV")
    np.testing.assert_allclose(temperature[0], 150.0)   # (100 + 200) / 2, not pulled towards zero
    np.testing.assert_allclose(temperature[-1], 750.0)
    co = smoothed.get_channel("F1 CO PV")
    assert np.isnan(co[:3]).all()                        # windows with no recorded sample stay NaN
    np.testing.assert_allclose(co[3:], 1.0)              # the NaN gap does not eat the first recorded values


def test_resample_skips_nan_samples():
    resampled = make_lv().resample(120.0, method="mean")
    temperature = resampled.get_channel("Reactor T PV")
    np.testing.assert_allclose(temperature, [250.0, 650.0])
    co = resampled.get_channel("F1 CO PV")
    assert np.isnan(co[0]) and co[1] == 1.0
    np.testing.assert_allclose(resampled.tos, [60.0, 180.0])
    assert make_lv().resample(120.0, method="last").get_channel("Reactor T PV").tolist() == [400.0, 800.0]


def test_rename_species_is_mirrored_into_the_dataset_attrs():
    renamed = make_lv().rename_species()
    assert renamed.ds.attrs["species_convention"] == renamed.metadata["species_convention"]
