"""Kernels: simple, parametrised lineshapes over plain arrays.

A lineshape is the shape of a line or distribution along one axis: a
spectral band, an XRD reflection, a chromatographic peak, a resonance curve,
a size distribution, a probability density (an area-scaled kernel with area 1
is one; the Gaussian is the normal, the Lorentzian the Cauchy distribution).
This module is the one home for every lineshape the parser and the analysis
code fit or draw, whatever the technique or axis. Nothing here
fits anything: the fits live with their technique
(:func:`phd_parser.spectroscopy.xps.fitting.fit_peaks`,
:func:`phd_parser.xrd.analysis.fit_reflection`) and evaluate these shapes.

Two families, told apart by the scale parameter:

**Height-scaled**, ``f(x, center, fwhm, height)``: :func:`gaussian`,
:func:`lorentzian` and :func:`pseudo_voigt` (the sum of the two with
Lorentzian fraction ``eta``); :func:`pseudo_voigt_area` gives the area.
Used for XRD reflections and mock spectra. Any of them, or
``functools.partial(pseudo_voigt, eta=0.4)``, is a valid kernel for a
generic multi-peak fit taking ``f(x, center, width, height)``.

**The Lorentzian fraction has two conventions.** ``eta`` weighs the two parts
at equal *height* by default (CasaXPS ``SGL``, this module's history) or at
equal *area* with ``fraction="area"`` (most diffraction literature, the
Thompson-Cox-Hastings formula). :func:`eta_to_area_fraction` and
:func:`eta_to_height_fraction` convert; the curve is the same, only the
number differs (height 0.57 is area 0.67).

**The Voigt profile** itself, the convolution of a Gaussian and a Lorentzian,
is :func:`voigt` (area-scaled, two widths), with :func:`voigt_fwhm` and
:func:`voigt_to_pseudo_voigt` relating it to the one-width approximations.

**Two asymmetric distributions** outside XPS, both area-scaled:
:func:`exponentially_modified_gaussian` (a Gaussian convolved with an
exponential decay, the tailing peak of chromatography) and :func:`log_normal`
(a quantity whose logarithm is normally distributed, the usual particle- and
pore-size distribution).

**Area-scaled**, ``f(x, center, fwhm, area, ...)``, written the way CasaXPS
writes them: :func:`gaussian_lorentzian_product` (``GL(m)``),
:func:`gaussian_lorentzian_sum` (``SGL(m)``, the area-scaled pseudo-Voigt)
and :func:`lorentzian_asymmetric` (``LA(alpha, beta, n)``). :class:`Lineshape`
holds one of them with its parameters and :func:`parse_lineshape` reads the
CasaXPS notation (``"GL(30)"``, ``"LA(1.53,243)"``).

Every ``fwhm`` is the true full width at half maximum of the curve, the
number CasaXPS reports (for ``SGL`` the shape parameter and the true width
coincide). Every ``area`` is the integrated area over all ``x``, exact
(closed forms, no numerical integration). ``center`` is the maximum of the
symmetric shapes and the Lorentzian centre of ``LA``.
"""

from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.signal import fftconvolve
from scipy.special import erfc, erfcx, gammaln, voigt_profile

__all__ = [
    "gaussian", "lorentzian", "pseudo_voigt", "pseudo_voigt_area",
    "EtaFraction", "eta_to_area_fraction", "eta_to_height_fraction",
    "voigt", "voigt_fwhm", "voigt_to_pseudo_voigt",
    "exponentially_modified_gaussian", "log_normal",
    "gaussian_lorentzian_sum", "gaussian_lorentzian_product", "lorentzian_asymmetric",
    "Lineshape", "LineshapeKind", "parse_lineshape",
    "GAUSSIAN_UNIT_AREA", "LORENTZIAN_UNIT_AREA", "DEFAULT_GAUSSIAN_WIDTH_SCALE",
]

FOUR_LN2 = 4.0 * math.log(2.0)
GAUSSIAN_UNIT_AREA = math.sqrt(math.pi / FOUR_LN2)  # area of exp(-4 ln2 u^2), unit height, unit FWHM
LORENTZIAN_UNIT_AREA = math.pi / 2.0  # area of 1 / (1 + 4 u^2), unit height, unit FWHM
DEFAULT_GAUSSIAN_WIDTH_SCALE = 500.0

LineshapeKind = Literal["GL", "SGL", "LA"]
EtaFraction = Literal["height", "area"]


# ----------------------------------------------------------------
# Height-scaled kernels
# ----------------------------------------------------------------


def gaussian(
    x: npt.NDArray[np.float64], center: float, fwhm: float, height: float
) -> npt.NDArray[np.float64]:
    """Gaussian line of a given full width at half maximum and height.

    Parameters
    ----------
    x : ndarray of float
        Points to evaluate at.
    center : float
        Position of the maximum.
    fwhm : float
        Full width at half maximum, in the units of ``x``.
    height : float
        Value at the maximum.

    Returns
    -------
    ndarray of float
        The line at ``x``.
    """
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def lorentzian(
    x: npt.NDArray[np.float64], center: float, fwhm: float, height: float
) -> npt.NDArray[np.float64]:
    """Lorentzian line of a given full width at half maximum and height.

    Parameters
    ----------
    x : ndarray of float
        Points to evaluate at.
    center : float
        Position of the maximum.
    fwhm : float
        Full width at half maximum, in the units of ``x``.
    height : float
        Value at the maximum.

    Returns
    -------
    ndarray of float
        The line at ``x``.
    """
    half_width = fwhm / 2.0
    return height / (1.0 + ((x - center) / half_width) ** 2)


def eta_to_area_fraction(eta: float) -> float:
    """Convert a Lorentzian fraction weighted at equal height into one weighted at equal area.

    Parameters
    ----------
    eta : float
        Height-weighted Lorentzian fraction, 0 to 1 (``pseudo_voigt``'s default, CasaXPS ``SGL``).

    Returns
    -------
    float
        ``eta L / (eta L + (1 - eta) G)``, with ``L = pi / 2`` and ``G = sqrt(pi / (4 ln 2))``
        the areas of the unit-height, unit-width Lorentzian and Gaussian.
    """
    eta = _check_fraction(eta)
    lorentzian_part = eta * LORENTZIAN_UNIT_AREA
    return float(lorentzian_part / (lorentzian_part + (1.0 - eta) * GAUSSIAN_UNIT_AREA))


def eta_to_height_fraction(eta_area: float) -> float:
    """Convert a Lorentzian fraction weighted at equal area into one weighted at equal height.

    Parameters
    ----------
    eta_area : float
        Area-weighted Lorentzian fraction, 0 to 1 (diffraction literature, Thompson-Cox-Hastings).

    Returns
    -------
    float
        ``eta_area G / (eta_area G + (1 - eta_area) L)``, the inverse of :func:`eta_to_area_fraction`.
    """
    eta_area = _check_fraction(eta_area)
    gaussian_part = eta_area * GAUSSIAN_UNIT_AREA
    return float(gaussian_part / (gaussian_part + (1.0 - eta_area) * LORENTZIAN_UNIT_AREA))


def _check_fraction(eta: float) -> float:
    eta = float(eta)
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"A Lorentzian fraction must lie in [0, 1], got {eta}")
    return eta


def _height_eta(eta: float, fraction: EtaFraction) -> float:
    """The height-weighted fraction for either convention."""
    if fraction == "height":
        return _check_fraction(eta)
    if fraction == "area":
        return eta_to_height_fraction(eta)
    raise ValueError(f"fraction must be 'height' or 'area', got {fraction!r}")


def pseudo_voigt(
    x: npt.NDArray[np.float64],
    center: float,
    fwhm: float,
    height: float,
    eta: float = 0.5,
    *,
    fraction: EtaFraction = "height",
) -> npt.NDArray[np.float64]:
    """Pseudo-Voigt line: a Lorentzian and a Gaussian of one width, scaled to one height.

    ``height * [e L1(x) + (1 - e) G1(x)]`` with ``L1``, ``G1`` of unit height
    and ``e`` the height-weighted fraction. With ``fraction="area"`` the
    given ``eta`` is read as the share of the *area* that is Lorentzian and
    converted first (:func:`eta_to_height_fraction`); the curve for a given
    shape is the same either way, only the number that names it differs.

    Parameters
    ----------
    x : ndarray of float
        Points to evaluate at.
    center : float
        Position of the maximum.
    fwhm : float
        Full width at half maximum shared by both parts, in the units of ``x``.
    height : float
        Value at the maximum.
    eta : float, optional
        Lorentzian fraction, 0 (Gaussian) to 1 (Lorentzian); default 0.5.
    fraction : {"height", "area"}, optional
        How ``eta`` weighs the two parts (default ``"height"``).

    Returns
    -------
    ndarray of float
        The line at ``x``.
    """
    weight = _height_eta(eta, fraction)
    return weight * lorentzian(x, center, fwhm, height) + (1.0 - weight) * gaussian(x, center, fwhm, height)


def pseudo_voigt_area(fwhm: float, height: float, eta: float, *, fraction: EtaFraction = "height") -> float:
    """Integrated area of a pseudo-Voigt line.

    Parameters
    ----------
    fwhm : float
        Full width at half maximum.
    height : float
        Value at the maximum.
    eta : float
        Lorentzian fraction.
    fraction : {"height", "area"}, optional
        How ``eta`` weighs the two parts (default ``"height"``), see :func:`pseudo_voigt`.

    Returns
    -------
    float
        ``height * fwhm * (e * pi / 2 + (1 - e) * sqrt(pi / (4 ln 2)))`` with ``e`` the height-weighted fraction.
    """
    weight = _height_eta(eta, fraction)
    return float(height * fwhm * (weight * LORENTZIAN_UNIT_AREA + (1.0 - weight) * GAUSSIAN_UNIT_AREA))


def voigt(
    x: npt.NDArray[np.float64],
    center: float,
    gaussian_fwhm: float,
    lorentzian_fwhm: float,
    area: float,
) -> npt.NDArray[np.float64]:
    """Voigt profile: a Gaussian convolved with a Lorentzian, scaled to its area.

    ``V = G ⊛ L`` has no closed form; it is evaluated through the Faddeeva
    function (``scipy.special.voigt_profile``). The convolution keeps the
    area, so ``area`` is exact. Unlike the one-width approximations it takes
    the two physical widths; its own FWHM follows from them (:func:`voigt_fwhm`).
    Either width may be zero (pure Lorentzian or pure Gaussian).

    Parameters
    ----------
    x : ndarray of float
        Points to evaluate at.
    center : float
        Position of the maximum.
    gaussian_fwhm : float
        FWHM of the Gaussian part (instrument, inhomogeneous broadening).
    lorentzian_fwhm : float
        FWHM of the Lorentzian part (lifetime broadening).
    area : float
        Integrated area over all ``x``.

    Returns
    -------
    ndarray of float
        The profile at ``x``.

    Raises
    ------
    ValueError
        If a width is negative or both are zero.
    """
    if gaussian_fwhm < 0 or lorentzian_fwhm < 0 or (gaussian_fwhm == 0 and lorentzian_fwhm == 0):
        raise ValueError(f"Voigt widths must be non-negative and not both zero, got {gaussian_fwhm}, {lorentzian_fwhm}")
    sigma = gaussian_fwhm / math.sqrt(2.0 * FOUR_LN2)
    gamma = lorentzian_fwhm / 2.0
    return area * voigt_profile(np.asarray(x, dtype=float) - center, sigma, gamma)


def voigt_fwhm(gaussian_fwhm: float, lorentzian_fwhm: float) -> float:
    """FWHM of a Voigt profile from its two widths (Olivero & Longbothum 1977).

    Parameters
    ----------
    gaussian_fwhm, lorentzian_fwhm : float
        The widths of the two parts.

    Returns
    -------
    float
        ``0.5346 wL + sqrt(0.2166 wL^2 + wG^2)``, accurate to about 0.02 %.
    """
    return float(0.5346 * lorentzian_fwhm + math.sqrt(0.2166 * lorentzian_fwhm**2 + gaussian_fwhm**2))


def voigt_to_pseudo_voigt(
    gaussian_fwhm: float, lorentzian_fwhm: float, *, fraction: EtaFraction = "height"
) -> Tuple[float, float]:
    """The pseudo-Voigt that approximates a given Voigt: its width and Lorentzian fraction.

    Thompson, Cox & Hastings (1987): the width is
    ``f = (wG^5 + 2.69269 wG^4 wL + 2.42843 wG^3 wL^2 + 4.47163 wG^2 wL^3 + 0.07842 wG wL^4 + wL^5)^(1/5)``
    and the area-weighted fraction ``1.36603 q - 0.47719 q^2 + 0.11116 q^3``
    with ``q = wL / f``. The result agrees with the Voigt to about 1 % of its
    height.

    Parameters
    ----------
    gaussian_fwhm, lorentzian_fwhm : float
        The widths of the Voigt's two parts.
    fraction : {"height", "area"}, optional
        Convention of the returned fraction (default ``"height"``, what
        :func:`pseudo_voigt` takes by default).

    Returns
    -------
    tuple of float
        ``(fwhm, eta)``.
    """
    wg, wl = float(gaussian_fwhm), float(lorentzian_fwhm)
    width = (wg**5 + 2.69269 * wg**4 * wl + 2.42843 * wg**3 * wl**2 + 4.47163 * wg**2 * wl**3
             + 0.07842 * wg * wl**4 + wl**5) ** 0.2
    q = wl / width
    eta_area = min(max(1.36603 * q - 0.47719 * q**2 + 0.11116 * q**3, 0.0), 1.0)
    if fraction == "area":
        return float(width), float(eta_area)
    if fraction == "height":
        return float(width), eta_to_height_fraction(eta_area)
    raise ValueError(f"fraction must be 'height' or 'area', got {fraction!r}")


def exponentially_modified_gaussian(
    x: npt.NDArray[np.float64],
    center: float,
    gaussian_fwhm: float,
    tau: float,
    area: float,
) -> npt.NDArray[np.float64]:
    """Exponentially modified Gaussian (EMG): a Gaussian convolved with an exponential decay.

    The shape of a peak that is broadened symmetrically (diffusion, the
    instrument) and then dragged out by a first-order process with time
    constant ``tau`` (slow desorption, a dead volume, a detector response):
    the classic tailing peak of chromatography. In statistics it is the
    density of the sum of a normal and an exponential random variable. With
    ``mu = center`` and ``sigma`` from ``gaussian_fwhm``::

        EMG(x) = A / (2 tau) exp(sigma^2 / (2 tau^2) - (x - mu) / tau)
                 erfc[(sigma / tau - (x - mu) / sigma) / sqrt(2)]

    evaluated through ``erfcx`` on the far side so that it neither overflows
    nor underflows. A negative ``tau`` mirrors the shape (a fronting peak);
    ``tau = 0`` is the Gaussian. Mean ``mu + tau``, variance
    ``sigma^2 + tau^2``; the maximum lies between ``mu`` and the mean, and
    ``gaussian_fwhm`` is the width of the Gaussian part, not of the curve.

    Parameters
    ----------
    x : ndarray of float
        Points to evaluate at.
    center : float
        Centre ``mu`` of the Gaussian part (not the maximum of the curve).
    gaussian_fwhm : float
        FWHM of the Gaussian part, ``2 sqrt(2 ln 2) sigma``.
    tau : float
        Time constant of the exponential, in the units of ``x``; positive tails
        to high ``x``, negative to low ``x``.
    area : float
        Integrated area over all ``x``.

    Returns
    -------
    ndarray of float
        The curve at ``x``.

    Raises
    ------
    ValueError
        If ``gaussian_fwhm`` is not positive.
    """
    if gaussian_fwhm <= 0:
        raise ValueError(f"gaussian_fwhm must be positive, got {gaussian_fwhm}")
    sigma = gaussian_fwhm / math.sqrt(2.0 * FOUR_LN2)
    x = np.asarray(x, dtype=float)
    if tau == 0:
        return area * np.exp(-0.5 * ((x - center) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))
    if tau < 0:
        return exponentially_modified_gaussian(2.0 * center - x, center, gaussian_fwhm, -tau, area)
    shifted = (x - center) / sigma
    z = (sigma / tau - shifted) / math.sqrt(2.0)
    out = np.empty_like(shifted)
    left = z >= 0  # erfc small or moderate: the erfcx form is safe
    out[left] = np.exp(-0.5 * shifted[left] ** 2) * erfcx(z[left])
    right = ~left  # deep in the tail: the exponent is negative there, the direct form is safe
    out[right] = np.exp(0.5 * (sigma / tau) ** 2 - shifted[right] * sigma / tau) * erfc(z[right])
    return area / (2.0 * tau) * out


def log_normal(
    x: npt.NDArray[np.float64],
    median: float,
    sigma_ln: float,
    area: float,
) -> npt.NDArray[np.float64]:
    """Log-normal distribution: ``ln x`` is normally distributed.

    The shape of a positive quantity built up by many small *multiplicative*
    steps (grinding, growth, sintering), as the Gaussian is for additive
    ones: the usual particle-, crystallite- and pore-size distribution::

        LN(x) = A / (x sigma_ln sqrt(2 pi)) exp[-(ln x - ln median)^2 / (2 sigma_ln^2)]

    for ``x > 0`` and zero elsewhere. Half the area lies below ``median``;
    the maximum is at ``median exp(-sigma_ln^2)``, the mean at
    ``median exp(sigma_ln^2 / 2)``; the tail is to large ``x``. On a
    logarithmic axis it is a Gaussian of standard deviation ``sigma_ln``.

    Parameters
    ----------
    x : ndarray of float
        Points to evaluate at (only ``x > 0`` carries weight).
    median : float
        The median, ``exp(mu)`` of the underlying normal distribution.
    sigma_ln : float
        Standard deviation of ``ln x``, dimensionless (geometric standard deviation ``exp(sigma_ln)``).
    area : float
        Integrated area over all ``x``.

    Returns
    -------
    ndarray of float
        The curve at ``x``.

    Raises
    ------
    ValueError
        If ``median`` or ``sigma_ln`` is not positive.
    """
    if median <= 0 or sigma_ln <= 0:
        raise ValueError(f"median and sigma_ln must be positive, got {median}, {sigma_ln}")
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    positive = x > 0
    xp = x[positive]
    out[positive] = np.exp(-0.5 * (np.log(xp / median) / sigma_ln) ** 2) / (xp * sigma_ln * math.sqrt(2.0 * math.pi))
    return area * out



# ----------------------------------------------------------------
# Area-scaled kernels (CasaXPS notation)
# ----------------------------------------------------------------


def gaussian_lorentzian_sum(
    x: npt.NDArray[np.float64],
    center: float,
    fwhm: float,
    area: float,
    lorentzian_fraction: float,
    *,
    fraction: EtaFraction = "height",
) -> npt.NDArray[np.float64]:
    """CasaXPS ``SGL(m)``: the weighted sum of a Gaussian and a Lorentzian of the same width.

    ``SGL(x) = (1 - m) exp(-4 ln2 u^2) + m / (1 + 4 u^2)`` with
    ``u = (x - E) / F``; CasaXPS writes ``m`` in percent, here it is the
    fraction ``0..1``. ``F`` is the full width at half maximum of both parts
    and of the sum.

    Parameters
    ----------
    x : numpy.ndarray
        Points at which to evaluate (binding energy in eV for XPS).
    center : float
        Peak centre ``E``.
    fwhm : float
        Width ``F``.
    area : float
        Integrated area over all energies.
    lorentzian_fraction : float
        Lorentzian weight ``m`` between 0 (Gaussian) and 1 (Lorentzian).
    fraction : {"height", "area"}, optional
        ``"height"`` (default) is CasaXPS's meaning of ``m``; ``"area"`` reads
        it as the Lorentzian share of the area, see :func:`pseudo_voigt`.

    Returns
    -------
    numpy.ndarray
        The curve, same shape as ``x``.
    """
    reduced = (np.asarray(x, dtype=float) - center) / fwhm
    weight = _height_eta(lorentzian_fraction, fraction)
    shape = (1.0 - weight) * np.exp(-FOUR_LN2 * reduced**2) + weight / (1.0 + 4.0 * reduced**2)
    unit_area = fwhm * ((1.0 - weight) * GAUSSIAN_UNIT_AREA + weight * LORENTZIAN_UNIT_AREA)
    return area / unit_area * shape


def _gaussian_lorentzian_product_unit_area(lorentzian_fraction: float) -> float:
    """Area of the unit-height, unit-width product ``exp(-4 ln2 (1-m) u^2) / (1 + 4 m u^2)``.

    Closed form: ``∫ exp(-a u^2) / (1 + b u^2) du = π / sqrt(b) · exp(a/b) erfc(sqrt(a/b))``
    with ``a = 4 ln2 (1 - m)`` and ``b = 4 m``; ``erfcx`` keeps it finite for small ``m``.
    """
    weight = float(lorentzian_fraction)
    if weight <= 0.0:
        return GAUSSIAN_UNIT_AREA
    a = FOUR_LN2 * (1.0 - weight)
    b = 4.0 * weight
    return math.pi / math.sqrt(b) * float(erfcx(math.sqrt(a / b)))


@lru_cache(maxsize=256)
def _gaussian_lorentzian_product_width_factor(lorentzian_fraction: float) -> float:
    """True FWHM of ``GL(m)`` in units of its shape parameter ``F`` (1 for m = 0 and m = 1)."""
    from scipy.optimize import brentq

    m = lorentzian_fraction
    half = brentq(lambda u: math.exp(-FOUR_LN2 * (1.0 - m) * u * u) / (1.0 + 4.0 * m * u * u) - 0.5, 0.0, 1.0)
    return 2.0 * half


def gaussian_lorentzian_product(
    x: npt.NDArray[np.float64],
    center: float,
    fwhm: float,
    area: float,
    lorentzian_fraction: float,
) -> npt.NDArray[np.float64]:
    """CasaXPS ``GL(m)``, the default lineshape: the product of a Gaussian and a Lorentzian.

    ``GL(x) = exp(-4 ln2 (1 - m) u^2) / (1 + 4 m u^2)`` with ``u = (x - E) / F``;
    CasaXPS writes ``m`` in percent, here it is the fraction ``0..1``. The
    ``fwhm`` argument is the true full width at half maximum of the curve,
    the number CasaXPS reports in its component table; the shape parameter
    ``F`` is derived from it (``GL(30)``: ``F = 1.04 fwhm``). Checked on 2026-10-18
    against CasaXPS exports: same curve to 0.03 % of the peak height. The area
    is taken from the closed form of the integral, so ``area`` is exact.

    Parameters
    ----------
    x : numpy.ndarray
        Points at which to evaluate (binding energy in eV for XPS).
    center : float
        Peak centre ``E``.
    fwhm : float
        Full width at half maximum.
    area : float
        Integrated area over all energies.
    lorentzian_fraction : float
        Lorentzian mixing ``m`` between 0 (Gaussian) and 1 (Lorentzian).

    Returns
    -------
    numpy.ndarray
        The curve, same shape as ``x``.
    """
    fwhm = fwhm / _gaussian_lorentzian_product_width_factor(float(lorentzian_fraction))   # true FWHM -> shape parameter F
    reduced = (np.asarray(x, dtype=float) - center) / fwhm
    weight = float(lorentzian_fraction)
    shape = np.exp(-FOUR_LN2 * (1.0 - weight) * reduced**2) / (1.0 + 4.0 * weight * reduced**2)
    return area / (fwhm * _gaussian_lorentzian_product_unit_area(weight)) * shape


def _half_area_of_lorentzian_power(exponent: float) -> float:
    """``∫_0^∞ (1 + 4 u^2)^-a du = sqrt(π) Γ(a - 1/2) / (4 Γ(a))``, finite for ``a > 1/2``."""
    if exponent <= 0.5:
        raise ValueError(f"A Lorentzian exponent must exceed 0.5 for a finite area, got {exponent}")
    return math.sqrt(math.pi) / 4.0 * math.exp(gammaln(exponent - 0.5) - gammaln(exponent))


def _asymmetric_lorentzian_unit_area(
    reduced: npt.NDArray[np.float64], alpha: float, beta: float, gaussian_fwhm: float
) -> npt.NDArray[np.float64]:
    """Unit-area ``LA`` shape in the reduced coordinate ``u = (x - E) / w``.

    ``(1 + 4 u^2)^-alpha`` for ``u > 0`` (high binding energy side),
    ``(1 + 4 u^2)^-beta`` for ``u <= 0``, convolved with a unit-area Gaussian
    of full width ``gaussian_fwhm`` (in units of ``w``) when that is not zero.
    """
    reduced = np.asarray(reduced, dtype=float)
    total_area = _half_area_of_lorentzian_power(alpha) + _half_area_of_lorentzian_power(beta)

    def base(u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        exponent = np.where(u > 0.0, alpha, beta)
        return np.power(1.0 + 4.0 * u**2, -exponent)

    if gaussian_fwhm <= 0.0:
        return base(reduced) / total_area
    sigma = gaussian_fwhm / math.sqrt(2.0 * FOUR_LN2)
    step = min(1.0, gaussian_fwhm) / 50.0
    margin = 6.0 * sigma + step
    low = (float(reduced.min()) if reduced.size else 0.0) - margin
    high = (float(reduced.max()) if reduced.size else 0.0) + margin
    if (high - low) / step > 400_000:  # a very wide window at a very narrow width: coarsen rather than stall
        step = (high - low) / 400_000
    # a grid through u = 0 keeps a symmetric shape exactly symmetric after interpolation
    first, last = math.floor(low / step), math.ceil(high / step)
    grid = step * np.arange(first, last + 1)
    half_kernel = int(math.ceil(6.0 * sigma / step))
    kernel_axis = step * np.arange(-half_kernel, half_kernel + 1)
    kernel = np.exp(-0.5 * (kernel_axis / sigma) ** 2)
    kernel /= kernel.sum()
    convolved = fftconvolve(base(grid), kernel, mode="same")
    return np.interp(reduced, grid, convolved) / total_area


def _fwhm_of_curve(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    """Full width at half maximum of a sampled single peak, crossings linearly interpolated."""
    index_max = int(np.argmax(y))
    half = 0.5 * float(y[index_max])
    below_left = np.nonzero(y[:index_max] < half)[0]
    below_right = np.nonzero(y[index_max:] < half)[0]
    if below_left.size == 0 or below_right.size == 0:
        raise ValueError("The sampled curve does not fall to half maximum on both sides")
    i = int(below_left[-1])
    left = x[i] + (half - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
    j = index_max + int(below_right[0])
    right = x[j - 1] + (half - y[j - 1]) * (x[j] - x[j - 1]) / (y[j] - y[j - 1])
    return float(right - left)


@functools.lru_cache(maxsize=256)
def _asymmetric_lorentzian_width_factor(alpha: float, beta: float, gaussian_fwhm: float) -> float:
    """FWHM of the unit-width ``LA`` shape, so that ``w = fwhm / factor`` gives a peak of width ``fwhm``."""
    reduced = np.linspace(-20.0, 20.0, 80_001)
    return _fwhm_of_curve(reduced, _asymmetric_lorentzian_unit_area(reduced, alpha, beta, gaussian_fwhm))


def lorentzian_asymmetric(
    x: npt.NDArray[np.float64],
    center: float,
    fwhm: float,
    area: float,
    alpha: float,
    beta: float,
    gaussian_width_index: float,
    gaussian_width_scale: float = DEFAULT_GAUSSIAN_WIDTH_SCALE,
) -> npt.NDArray[np.float64]:
    """CasaXPS ``LA(alpha, beta, n)``: an asymmetric Lorentzian convolved with a Gaussian.

    The Lorentzian ``L(u) = 1 / (1 + 4 u^2)``, ``u = (x - E) / w``, is raised
    to a different power on the two sides of its centre and then convolved
    with a Gaussian::

        LA(x) = A · [ L^alpha (x > E) ∪ L^beta (x <= E) ] ⊛ G_n

    with ``x`` the binding energy. **alpha acts on the high binding energy
    side** (``x > E``, lower kinetic energy), **beta on the low binding
    energy side**; a smaller exponent means a longer tail, so ``LA(1.1, 5, n)``
    is the metal-like shape with its tail towards higher binding energy.
    CasaXPS defines the shape on its kinetic energy axis with alpha for
    ``x <= E``, which is the same side. The two-argument form ``LA(a, n)``
    is the symmetric case ``alpha = beta = a``.

    The Gaussian width index ``n`` is CasaXPS's integer ``0..499``
    "characterising" the convolution width, whose mapping to eV CasaXPS does
    not publish. **This implementation maps it relative to the Lorentzian
    width:** the Gaussian has full width ``w · n / gaussian_width_scale``
    (default scale 500, so ``n = 250`` convolves with a Gaussian half as wide
    as the Lorentzian and ``n = 0`` is no convolution). That keeps the shape
    scale-invariant in ``w``. It is a stated choice, not CasaXPS's internal
    constant; change ``gaussian_width_scale`` if a calibration against CasaXPS
    output ever gives a better value.

    ``fwhm`` is the full width at half maximum of the final curve (measured
    numerically once per ``(alpha, beta, n)`` and cached), not the Lorentzian
    width ``w`` before the exponents and the convolution. ``center`` is the
    Lorentzian centre ``E``; for ``alpha != beta`` the maximum of the curve
    sits slightly off it. The area is exact: each half of ``L^a`` integrates
    to ``sqrt(π) Γ(a - 1/2) / (4 Γ(a)) · w`` and the convolution keeps it.

    Parameters
    ----------
    x : numpy.ndarray
        Points at which to evaluate (binding energy in eV for XPS).
    center : float
        Lorentzian centre ``E``.
    fwhm : float
        Full width at half maximum of the resulting peak.
    area : float
        Integrated area over all energies.
    alpha : float
        Exponent on the high binding energy side, greater than 0.5.
    beta : float
        Exponent on the low binding energy side, greater than 0.5.
    gaussian_width_index : float
        CasaXPS's ``n``; the Gaussian FWHM is ``w · n / gaussian_width_scale``.
    gaussian_width_scale : float, optional
        Index per unit of Lorentzian width (default is 500).

    Returns
    -------
    numpy.ndarray
        The curve, same shape as ``x``.

    Raises
    ------
    ValueError
        If an exponent is 0.5 or less (the area diverges).
    """
    gaussian_fwhm = float(gaussian_width_index) / float(gaussian_width_scale)
    factor = _asymmetric_lorentzian_width_factor(float(alpha), float(beta), gaussian_fwhm)
    width = fwhm / factor
    reduced = (np.asarray(x, dtype=float) - center) / width
    return area / width * _asymmetric_lorentzian_unit_area(reduced, float(alpha), float(beta), gaussian_fwhm)


@dataclass(frozen=True)
class Lineshape:
    """A CasaXPS lineshape: its kind and parameters, evaluated with position, width and area.

    Attributes
    ----------
    kind : {"GL", "SGL", "LA"}
        Product (CasaXPS's default), sum, or asymmetric Lorentzian.
    lorentzian_fraction : float or None
        ``m / 100`` for ``GL`` and ``SGL``.
    alpha, beta : float or None
        The exponents of ``LA`` (high and low binding energy side).
    gaussian_width_index : float or None
        The Gaussian width index ``n`` of ``LA``.
    gaussian_width_scale : float
        How ``n`` maps to the Gaussian width, see :func:`lorentzian_asymmetric`.
    """

    kind: LineshapeKind
    lorentzian_fraction: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gaussian_width_index: Optional[float] = None
    gaussian_width_scale: float = DEFAULT_GAUSSIAN_WIDTH_SCALE

    def __post_init__(self) -> None:
        if self.kind in ("GL", "SGL"):
            if self.lorentzian_fraction is None or not 0.0 <= self.lorentzian_fraction <= 1.0:
                raise ValueError(f"{self.kind} needs a lorentzian_fraction in [0, 1], got {self.lorentzian_fraction}")
        elif self.kind == "LA":
            if self.alpha is None or self.beta is None or self.gaussian_width_index is None:
                raise ValueError("LA needs alpha, beta and gaussian_width_index")
            if self.alpha <= 0.5 or self.beta <= 0.5:
                raise ValueError(f"LA exponents must exceed 0.5, got alpha={self.alpha}, beta={self.beta}")
            if self.gaussian_width_index < 0:
                raise ValueError(f"gaussian_width_index must not be negative, got {self.gaussian_width_index}")
        else:
            raise ValueError(f"Unknown lineshape kind {self.kind!r}; use 'GL', 'SGL' or 'LA'")

    def evaluate(
        self, x: npt.NDArray[np.float64], center: float, fwhm: float, area: float
    ) -> npt.NDArray[np.float64]:
        """Evaluate the lineshape.

        Parameters
        ----------
        x : numpy.ndarray
            Points at which to evaluate.
        center : float
            Centre.
        fwhm : float
            True full width at half maximum, for every kind.
        area : float
            Integrated area.

        Returns
        -------
        numpy.ndarray
            The curve, same shape as ``x``.
        """
        if self.kind == "GL":
            return gaussian_lorentzian_product(x, center, fwhm, area, self.lorentzian_fraction)  # type: ignore[arg-type]
        if self.kind == "SGL":
            return gaussian_lorentzian_sum(x, center, fwhm, area, self.lorentzian_fraction)  # type: ignore[arg-type]
        return lorentzian_asymmetric(
            x, center, fwhm, area, self.alpha, self.beta, self.gaussian_width_index, self.gaussian_width_scale  # type: ignore[arg-type]
        )

    @property
    def casaxps(self) -> str:
        """The CasaXPS notation, ``GL(30)``, ``SGL(30)``, ``LA(1.53,243)`` or ``LA(1.2,1.5,100)``."""
        if self.kind in ("GL", "SGL"):
            return f"{self.kind}({_format_number(100.0 * self.lorentzian_fraction)})"  # type: ignore[operator]
        parts = [self.alpha] if self.alpha == self.beta else [self.alpha, self.beta]
        parts.append(self.gaussian_width_index)
        return "LA(" + ",".join(_format_number(part) for part in parts) + ")"  # type: ignore[arg-type]

    def __str__(self) -> str:
        """The CasaXPS notation."""
        return self.casaxps


def _format_number(value: float) -> str:
    text = f"{float(value):.6g}"
    return text


_LINESHAPE_PATTERN = re.compile(r"^\s*(GL|SGL|LA)\s*\(\s*([^)]*?)\s*\)\s*$", re.IGNORECASE)


def parse_lineshape(text: Union[str, Lineshape]) -> Lineshape:
    """Read a CasaXPS lineshape string.

    Parameters
    ----------
    text : str or Lineshape
        ``"GL(30)"``, ``"SGL(30)"``, ``"LA(1.53,243)"`` (symmetric, ``alpha = beta``)
        or ``"LA(1.2,1.5,100)"``; a :class:`Lineshape` is returned as is. The
        one-number ``LA(50)`` of CasaXPS is refused, see the error message.

    Returns
    -------
    Lineshape
        The parsed lineshape.

    Raises
    ------
    ValueError
        If the text is not one of the three notations.
    """
    if isinstance(text, Lineshape):
        return text
    match = _LINESHAPE_PATTERN.match(str(text))
    if match is None:
        raise ValueError(f"Cannot read lineshape {text!r}; expected GL(m), SGL(m), LA(a,n) or LA(a,b,n)")
    kind = match.group(1).upper()
    try:
        numbers = [float(part) for part in match.group(2).split(",") if part.strip()]
    except ValueError as error:
        raise ValueError(f"Cannot read the numbers of lineshape {text!r}") from error
    if kind in ("GL", "SGL"):
        if len(numbers) != 1:
            raise ValueError(f"{kind} takes one number (the Lorentzian percentage), got {text!r}")
        return Lineshape(kind=kind, lorentzian_fraction=numbers[0] / 100.0)  # type: ignore[arg-type]
    if len(numbers) == 1:
        # CasaXPS accepts LA(m) with one number, but its manual gives no formula for it, and the
        # curves it draws for LA(50) are not any LA(a, a, n) of this module (closest LA(1.5,1.5,800),
        # 0.4 % off; compared on 2026-10-18 against CasaXPS exports). Refuse rather than guess.
        raise ValueError(f"{text!r}: the one-number form LA(m) is CasaXPS-internal and not reproduced here; "
                         "give LA(a,n) or LA(a,b,n). CasaXPS's LA(50) is closest to LA(1.5,800) in this module's convention.")
    if len(numbers) == 2:
        return Lineshape(kind="LA", alpha=numbers[0], beta=numbers[0], gaussian_width_index=numbers[1])
    if len(numbers) == 3:
        return Lineshape(kind="LA", alpha=numbers[0], beta=numbers[1], gaussian_width_index=numbers[2])
    raise ValueError(f"LA takes two or three numbers, got {text!r}")
