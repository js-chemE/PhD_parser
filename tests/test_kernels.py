"""Kernels: the height-scaled Gaussian, Lorentzian and pseudo-Voigt, and the CasaXPS GL, SGL and LA."""
import numpy as np
import pytest

from phd_parser import kernels
from phd_parser.kernels import (
    Lineshape,
    gaussian,
    gaussian_lorentzian_product,
    gaussian_lorentzian_sum,
    lorentzian,
    lorentzian_asymmetric,
    parse_lineshape,
    pseudo_voigt,
    pseudo_voigt_area,
)

FINE = np.linspace(-80.0, 80.0, 320_001)


def test_the_two_families_agree_sgl_is_the_area_scaled_pseudo_voigt():
    x = np.linspace(-10.0, 10.0, 2001)
    for eta in (0.0, 0.3, 1.0):
        height = 2.0
        area = pseudo_voigt_area(1.3, height, eta)
        np.testing.assert_allclose(gaussian_lorentzian_sum(x, 0.5, 1.3, area, eta), pseudo_voigt(x, 0.5, 1.3, height, eta), rtol=1e-12)


def test_pseudo_voigt_limits_and_width():
    # wide enough that the Lorentzian tails hold less than 0.2 % of the area
    x = np.linspace(-200.0, 200.0, 400001)
    np.testing.assert_allclose(pseudo_voigt(x, 0.0, 1.0, 2.0, eta=0.0), gaussian(x, 0.0, 1.0, 2.0))
    np.testing.assert_allclose(pseudo_voigt(x, 0.0, 1.0, 2.0, eta=1.0), lorentzian(x, 0.0, 1.0, 2.0))
    for eta in (0.0, 0.3, 1.0):
        y = pseudo_voigt(x, 0.0, 1.0, 2.0, eta=eta)
        assert y.max() == pytest.approx(2.0)
        above_half = x[y >= 1.0]
        assert above_half.max() - above_half.min() == pytest.approx(1.0, abs=1e-3)
        assert np.trapezoid(y, x) == pytest.approx(pseudo_voigt_area(1.0, 2.0, eta), rel=0.02)


@pytest.mark.parametrize("text", ["GL(30)", "SGL(30)", "LA(1.53,243)", "LA(1.2,1.5,100)", "GL(0)", "GL(100)"])
def test_every_lineshape_integrates_to_its_area(text):
    curve = parse_lineshape(text).evaluate(FINE, 0.0, 1.0, 2.5)
    # the Lorentzian tails beyond ±80 widths hold a few tenths of a percent at most
    assert np.trapezoid(curve, FINE) == pytest.approx(2.5, rel=5e-3)


def test_gaussian_lorentzian_forms_match_casaxps_definitions():
    reduced = FINE
    gaussian = np.exp(-4 * np.log(2) * reduced**2)
    lorentzian = 1 / (1 + 4 * reduced**2)
    factor = kernels._gaussian_lorentzian_product_width_factor(0.3)
    product = gaussian_lorentzian_product(FINE, 0.0, factor, 1.0, 0.3)    # fwhm argument = true width; F = 1
    total = gaussian_lorentzian_sum(FINE, 0.0, 1.0, 1.0, 0.3)
    expected_product = gaussian**0.7 / (1 + 4 * 0.3 * reduced**2)
    expected_sum = 0.7 * gaussian + 0.3 * lorentzian
    np.testing.assert_allclose(product / product.max(), expected_product, atol=1e-12)
    np.testing.assert_allclose(total / total.max(), expected_sum, atol=1e-12)
    # with shape parameter F = 1 the product is a little narrower than 1 (CasaXPS's definition)
    half = np.abs(FINE[total >= total.max() / 2])
    assert 2 * half.max() == pytest.approx(1.0, abs=1e-3)
    half = np.abs(FINE[product >= product.max() / 2])
    assert 2 * half.max() == pytest.approx(factor, abs=1e-3) and factor < 1.0


def test_lorentzian_asymmetric_is_normalised_and_tails_to_high_binding_energy():
    symmetric = lorentzian_asymmetric(FINE, 0.0, 1.0, 1.0, 1.53, 1.53, 243)
    assert np.trapezoid(symmetric, FINE) == pytest.approx(1.0, rel=2e-3)
    np.testing.assert_allclose(symmetric, symmetric[::-1], atol=1e-6)
    half = np.abs(FINE[symmetric >= symmetric.max() / 2])
    assert 2 * half.max() == pytest.approx(1.0, abs=2e-3)  # fwhm is the width of the final curve
    # alpha < beta: the smaller exponent, on the high binding energy side, gives the longer tail
    asymmetric = lorentzian_asymmetric(FINE, 0.0, 1.0, 1.0, 1.0, 3.0, 50)
    assert np.trapezoid(asymmetric, FINE) == pytest.approx(1.0, rel=1e-2)
    at = lambda x: float(np.interp(x, FINE, asymmetric))  # noqa: E731
    assert at(2.0) > 10 * at(-2.0)
    assert at(0.75) > at(-0.75)
    # no convolution for n = 0, and an exponent at or below 1/2 has no finite area
    bare = lorentzian_asymmetric(np.array([0.0, 0.5]), 0.0, 1.0, 1.0, 1.0, 1.0, 0)
    assert bare[1] == pytest.approx(bare[0] / 2)
    with pytest.raises(ValueError):
        lorentzian_asymmetric(FINE, 0.0, 1.0, 1.0, 0.5, 1.0, 0)


def test_parse_lineshape_round_trips_the_casaxps_notation():
    for text in ("GL(30)", "SGL(30)", "LA(1.53,243)", "LA(1.2,1.5,100)"):
        shape = parse_lineshape(text)
        assert shape.casaxps == text and str(shape) == text
        assert parse_lineshape(shape) is shape
    assert parse_lineshape("gl(30)") == Lineshape("GL", lorentzian_fraction=0.3)
    la = parse_lineshape("LA(1.53,243)")
    assert la.kind == "LA" and la.alpha == la.beta == 1.53 and la.gaussian_width_index == 243
    for bad in ("Voigt(30)", "GL(30,40)", "LA(1)", "GL(130)", "LA(0.4,1,10)"):
        with pytest.raises(ValueError):
            parse_lineshape(bad)


def test_one_number_la_is_refused_with_an_explanation():
    with pytest.raises(ValueError, match="CasaXPS-internal"):
        parse_lineshape("LA(50)")


def test_gl_fwhm_is_the_true_width():
    energy = np.linspace(-5, 5, 20001)
    for m in (0.0, 0.3, 0.7, 1.0):
        curve = gaussian_lorentzian_product(energy, 0.0, 1.6, 1.0, m)
        half = energy[curve >= curve.max() / 2]
        assert half.max() - half.min() == pytest.approx(1.6, abs=1e-3)
        if m < 1.0:     # a pure Lorentzian keeps ~10 % of its area beyond +-5 eV
            assert np.trapezoid(curve, energy) == pytest.approx(1.0, rel=1e-3)
