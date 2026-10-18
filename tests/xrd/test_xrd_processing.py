"""XRDData processing keeps the scan coordinates and never edits the caller's array."""
import numpy as np
import pandas as pd
import xarray as xr

from phd_parser.xrd import XRDData


def make_series():
    angle = np.linspace(10.0, 80.0, 141)
    values = np.outer([1.0, 2.0], np.exp(-((angle - 25.0) / 0.5) ** 2))
    da = xr.DataArray(
        values, dims=["scan", "angle"],
        coords={
            "angle": angle, "scan": [0, 1], "tos": ("scan", [0.0, 60.0]),
            "timestamp": ("scan", pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:01"])),
        },
    )
    return da, XRDData(da=da)


def test_processing_keeps_tos_and_timestamp_coordinates():
    _, xrd = make_series()
    for processed in (
        xrd.select_angle_range(20.0, 30.0),
        xrd.smooth_savgol(11, 3),
        xrd.smooth_gaussian(0.2),
        xrd.smooth_moving(5),
        xrd.sort(ascending=False),
    ):
        assert set(processed.da.coords) == {"angle", "scan", "tos", "timestamp"}
        np.testing.assert_allclose(processed.da.coords["tos"].values, [0.0, 60.0])
    assert xrd.select_angle_range(20.0, 30.0).angle.min() >= 20.0


def test_the_constructor_does_not_edit_the_input_array():
    da, xrd = make_series()
    assert "angle_units" not in da.attrs
    assert xrd.da.attrs["angle_units"] == "deg"


def test_d_spacing_and_q_vector_follow_bragg():
    _, xrd = make_series()
    wavelength = 1.5406
    theta = np.deg2rad(xrd.angle / 2.0)
    np.testing.assert_allclose(xrd.d_spacing(wavelength), wavelength / (2.0 * np.sin(theta)))
    np.testing.assert_allclose(xrd.q_vector(wavelength), 4.0 * np.pi * np.sin(theta) / wavelength)
