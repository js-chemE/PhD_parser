"""Reflection fitting and the Scherrer size, on a mock line and on the anatase (101) of the fixture."""
from pathlib import Path

import numpy as np
import pytest

from phd_parser.kernels import gaussian, pseudo_voigt, pseudo_voigt_area
from phd_parser.xrd import ReferenceCard, ReflectionFit, XRDData, fit_reflection, scherrer

HERE = Path(__file__).parent
FIXTURE = HERE / "bruker_SK0003_MoO3-12pct-TiO2_12-28deg.xy"
ANATASE = HERE / "PDF 78-2486_TiO2-Anatase.txt"


def test_fit_recovers_a_mock_line_on_a_sloped_background():
    angle = np.arange(20.0, 30.0, 0.01)
    truth = dict(center=25.0, fwhm=0.4, height=100.0, eta=0.3)
    values = pseudo_voigt(angle, **truth) + 5.0 + 0.5 * (angle - 25.0)
    rng = np.random.default_rng(0)
    values = values + rng.normal(0.0, 0.3, angle.size)
    fit = fit_reflection(angle, values, 25.05, window_deg=2.0)
    assert isinstance(fit, ReflectionFit)
    assert fit.position_deg == pytest.approx(25.0, abs=0.005)
    assert fit.fwhm_deg == pytest.approx(0.4, abs=0.01)
    assert fit.height == pytest.approx(100.0, rel=0.02)
    assert fit.eta == pytest.approx(0.3, abs=0.1)
    assert fit.background[0] == pytest.approx(5.0, abs=0.3)
    assert fit.background[1] == pytest.approx(0.5, abs=0.1)
    assert fit.area == pytest.approx(pseudo_voigt_area(0.4, 100.0, 0.3), rel=0.03)
    assert fit.n_points > 7 and fit.lineshape == "pseudo_voigt"
    assert set(fit.standard_errors) >= {"position_deg", "fwhm_deg", "height", "eta"}
    np.testing.assert_allclose(fit.model(angle), fit.line(angle) + fit.background_at(angle))
    assert fit.to_dict()["background"] == list(fit.background)


def test_fixed_lineshapes():
    angle = np.arange(20.0, 30.0, 0.01)
    values = gaussian(angle, 25.0, 0.5, 50.0) + 2.0
    assert fit_reflection(angle, values, 25.0, lineshape="gaussian").eta == 0.0
    assert fit_reflection(angle, values, 25.0, lineshape="lorentzian").eta == 1.0
    assert fit_reflection(angle, values, 25.0, lineshape="gaussian").fwhm_deg == pytest.approx(0.5, abs=1e-3)
    with pytest.raises(ValueError):
        fit_reflection(angle, values, 25.0, lineshape="voigt")
    with pytest.raises(ValueError):
        fit_reflection(angle, values, 25.0, window_deg=0.03)


def test_anatase_101_of_the_fixture():
    xrd = XRDData.from_bruker_xy(FIXTURE)
    card = ReferenceCard.from_file(ANATASE)
    reference = float(card.strongest(1)["two_theta_deg"].iloc[0])
    fit = xrd.fit_reflection(reference)
    assert fit.position_deg == pytest.approx(25.3, abs=0.1)
    assert 0.3 <= fit.fwhm_deg <= 1.0
    assert fit.height > 50.0
    assert 0.0 <= fit.eta <= 1.0
    assert fit.area > 0.0
    size_nm = xrd.scherrer_size(fit.position_deg, fit.fwhm_deg)
    assert 5.0 <= size_nm <= 30.0
    # the wavelength came from the anode in the header
    assert size_nm == pytest.approx(scherrer(fit.position_deg, fit.fwhm_deg, wavelength_angstrom=1.5406))
    # the fit follows the data
    window = xrd.select_angle_range(*fit.window_deg)
    residual = np.sqrt(np.mean((window.values - fit.model(window.angle)) ** 2))
    assert residual < 0.05 * fit.height


def test_fit_reflection_on_two_dimensional_data():
    angle = np.arange(20.0, 30.0, 0.01)
    values = np.vstack([gaussian(angle, 25.0, 0.5, 50.0), gaussian(angle, 25.2, 0.7, 20.0)])
    import xarray as xr

    xrd = XRDData(da=xr.DataArray(values, dims=["scan", "angle"], coords={"angle": angle, "scan": [0, 1]}))
    with pytest.raises(ValueError):
        xrd.fit_reflection(25.0)
    assert xrd.fit_reflection(25.0, scan=1).position_deg == pytest.approx(25.2, abs=1e-3)


def test_scherrer_formula_and_instrumental_correction():
    # K λ / (β cos θ): 0.9 * 1.5406 Å / (0.5° in rad * cos 12.65°) = 14.0 nm
    expected = 0.9 * 1.5406 / (np.deg2rad(0.5) * np.cos(np.deg2rad(25.3 / 2.0))) / 10.0
    assert scherrer(25.3, 0.5) == pytest.approx(expected)
    assert scherrer(25.3, 0.5, wavelength_angstrom=1.7890) == pytest.approx(expected * 1.7890 / 1.5406)
    assert scherrer(25.3, 0.5, shape_factor=1.0) == pytest.approx(expected / 0.9)
    corrected = scherrer(25.3, 0.5, instrumental_fwhm_deg=0.3)
    assert corrected == pytest.approx(scherrer(25.3, 0.4))
    with pytest.raises(ValueError):
        scherrer(25.3, 0.5, instrumental_fwhm_deg=0.5)
    with pytest.raises(ValueError):
        scherrer(25.3, 0.0)


def test_scherrer_size_wavelength_defaults():
    import xarray as xr

    angle = np.arange(20.0, 30.0, 0.01)
    da = xr.DataArray(gaussian(angle, 25.0, 0.5, 50.0), dims=["angle"], coords={"angle": angle})
    without = XRDData(da=da)
    assert without.scherrer_size(25.0, 0.5) == pytest.approx(scherrer(25.0, 0.5, wavelength_angstrom=1.5406))
    cobalt = XRDData(da=da.assign_attrs(wavelength_angstrom=1.7890))
    assert cobalt.scherrer_size(25.0, 0.5) == pytest.approx(scherrer(25.0, 0.5, wavelength_angstrom=1.7890))
    assert cobalt.scherrer_size(25.0, 0.5, wavelength_angstrom=1.5406) == pytest.approx(
        scherrer(25.0, 0.5, wavelength_angstrom=1.5406)
    )
