"""CasaXPS-style peak fitting of one XPS region: lineshapes, peak specifications and the fit.

The module replaces the curve-fitting step of CasaXPS so an analysis can be
done in Python end to end. It knows three lineshapes written the way
CasaXPS writes them (``GL(30)``, ``SGL(30)``, ``LA(1.53,243)``,
``LA(1.2,1.5,100)``), a :class:`PeakSpec` that describes one peak with its
initial guesses, bounds and the CasaXPS-style constraints to another peak
(fixed spin-orbit splitting, fixed area ratio, equal width), the
:func:`doublet` helper with the nominal splittings of :data:`DOUBLETS`, and
:func:`fit_peaks`, a bounded least-squares fit that returns a
:class:`FitResult`. :meth:`XPSData.fit_peaks <phd_parser.spectroscopy.xps.core.XPSData.fit_peaks>`
wraps all of it for a region.

Every lineshape is normalised so that its ``area`` argument is the integrated
area over all energies, in ``data_type`` units times eV. The ``position`` is
the centre parameter of the lineshape (the maximum for the symmetric ones,
the centre of the Lorentzian for the asymmetric one), as CasaXPS reports it.
The ``fwhm`` is CasaXPS's width parameter ``F`` for ``GL`` and ``SGL`` and
the full width at half maximum of the final curve for ``LA`` (see
:func:`lorentzian_asymmetric`).
"""

from __future__ import annotations

import dataclasses
import functools
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import fftconvolve
from scipy.special import erfcx, gammaln

FOUR_LN2 = 4.0 * math.log(2.0)
GAUSSIAN_UNIT_AREA = math.sqrt(math.pi / FOUR_LN2)  # area of exp(-4 ln2 u^2), unit height, unit FWHM
LORENTZIAN_UNIT_AREA = math.pi / 2.0  # area of 1 / (1 + 4 u^2), unit height, unit FWHM
DEFAULT_GAUSSIAN_WIDTH_SCALE = 500.0

LineshapeKind = Literal["GL", "SGL", "LA"]


# ----------------------------------------------------------------
# Lineshapes
# ----------------------------------------------------------------


def gaussian_lorentzian_sum(
    energy: npt.NDArray[np.float64],
    position: float,
    fwhm: float,
    area: float,
    lorentzian_fraction: float,
) -> npt.NDArray[np.float64]:
    """CasaXPS ``SGL(m)``: the weighted sum of a Gaussian and a Lorentzian of the same width.

    ``SGL(x) = (1 - m) exp(-4 ln2 u^2) + m / (1 + 4 u^2)`` with
    ``u = (x - E) / F``; CasaXPS writes ``m`` in percent, here it is the
    fraction ``0..1``. ``F`` is the full width at half maximum of both parts
    and of the sum.

    Parameters
    ----------
    energy : numpy.ndarray
        Binding energies in eV at which to evaluate.
    position : float
        Peak centre ``E`` in eV.
    fwhm : float
        Width ``F`` in eV.
    area : float
        Integrated area over all energies.
    lorentzian_fraction : float
        Lorentzian weight ``m`` between 0 (Gaussian) and 1 (Lorentzian).

    Returns
    -------
    numpy.ndarray
        The curve, same shape as ``energy``.
    """
    reduced = (np.asarray(energy, dtype=float) - position) / fwhm
    weight = float(lorentzian_fraction)
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


def gaussian_lorentzian_product(
    energy: npt.NDArray[np.float64],
    position: float,
    fwhm: float,
    area: float,
    lorentzian_fraction: float,
) -> npt.NDArray[np.float64]:
    """CasaXPS ``GL(m)``, the default lineshape: the product of a Gaussian and a Lorentzian.

    ``GL(x) = exp(-4 ln2 (1 - m) u^2) / (1 + 4 m u^2)`` with ``u = (x - E) / F``;
    CasaXPS writes ``m`` in percent, here it is the fraction ``0..1``. ``F``
    is CasaXPS's width parameter: it is the full width at half maximum only
    for ``m = 0`` and ``m = 1``, in between the product is a little narrower
    (``GL(30)`` has a FWHM of ``0.95 F``). The area is taken from the closed
    form of the integral, so ``area`` is exact.

    Parameters
    ----------
    energy : numpy.ndarray
        Binding energies in eV at which to evaluate.
    position : float
        Peak centre ``E`` in eV.
    fwhm : float
        Width parameter ``F`` in eV.
    area : float
        Integrated area over all energies.
    lorentzian_fraction : float
        Lorentzian mixing ``m`` between 0 (Gaussian) and 1 (Lorentzian).

    Returns
    -------
    numpy.ndarray
        The curve, same shape as ``energy``.
    """
    reduced = (np.asarray(energy, dtype=float) - position) / fwhm
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
    energy: npt.NDArray[np.float64],
    position: float,
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
    width ``w`` before the exponents and the convolution. ``position`` is the
    Lorentzian centre ``E``; for ``alpha != beta`` the maximum of the curve
    sits slightly off it. The area is exact: each half of ``L^a`` integrates
    to ``sqrt(π) Γ(a - 1/2) / (4 Γ(a)) · w`` and the convolution keeps it.

    Parameters
    ----------
    energy : numpy.ndarray
        Binding energies in eV at which to evaluate.
    position : float
        Lorentzian centre ``E`` in eV.
    fwhm : float
        Full width at half maximum of the resulting peak in eV.
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
        The curve, same shape as ``energy``.

    Raises
    ------
    ValueError
        If an exponent is 0.5 or less (the area diverges).
    """
    gaussian_fwhm = float(gaussian_width_index) / float(gaussian_width_scale)
    factor = _asymmetric_lorentzian_width_factor(float(alpha), float(beta), gaussian_fwhm)
    width = fwhm / factor
    reduced = (np.asarray(energy, dtype=float) - position) / width
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
        self, energy: npt.NDArray[np.float64], position: float, fwhm: float, area: float
    ) -> npt.NDArray[np.float64]:
        """Evaluate the lineshape.

        Parameters
        ----------
        energy : numpy.ndarray
            Binding energies in eV.
        position : float
            Centre in eV.
        fwhm : float
            Width in eV (see the module docstring for what it means per kind).
        area : float
            Integrated area.

        Returns
        -------
        numpy.ndarray
            The curve, same shape as ``energy``.
        """
        if self.kind == "GL":
            return gaussian_lorentzian_product(energy, position, fwhm, area, self.lorentzian_fraction)  # type: ignore[arg-type]
        if self.kind == "SGL":
            return gaussian_lorentzian_sum(energy, position, fwhm, area, self.lorentzian_fraction)  # type: ignore[arg-type]
        return lorentzian_asymmetric(
            energy, position, fwhm, area, self.alpha, self.beta, self.gaussian_width_index, self.gaussian_width_scale  # type: ignore[arg-type]
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
        or ``"LA(1.2,1.5,100)"``; a :class:`Lineshape` is returned as is.

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
    if len(numbers) == 2:
        return Lineshape(kind="LA", alpha=numbers[0], beta=numbers[0], gaussian_width_index=numbers[1])
    if len(numbers) == 3:
        return Lineshape(kind="LA", alpha=numbers[0], beta=numbers[1], gaussian_width_index=numbers[2])
    raise ValueError(f"LA takes two or three numbers, got {text!r}")


# ----------------------------------------------------------------
# Peak specification
# ----------------------------------------------------------------


@dataclass
class PeakSpec:
    """One peak to fit: initial guesses, bounds and CasaXPS-style constraints to another peak.

    Attributes
    ----------
    name : str
        Unique name, also what the constraints of other peaks refer to.
    position_eV : float
        Initial centre in eV.
    fwhm_eV : float
        Initial width in eV (default is 1.0).
    area : float or None
        Initial area; ``None`` (default) estimates it from the data at the
        initial position.
    lineshape : Lineshape or str
        CasaXPS notation or a :class:`Lineshape` (default is ``"GL(30)"``).
    position_bounds_eV : tuple of float or None
        Lower and upper bound of the centre; ``None`` (default) is ±1 eV
        around the guess.
    fwhm_bounds_eV : tuple of float
        Bounds of the width (default is 0.3 to 4 eV).
    area_bounds : tuple of float
        Bounds of the area (default is 0 to infinity).
    position_offset_from : tuple of (str, float) or None
        ``(other_name, delta_eV)``: the centre is ``other + delta_eV``
        (a fixed spin-orbit splitting or chemical shift), not fitted.
    area_ratio_to : tuple of (str, float) or None
        ``(other_name, ratio)``: the area is ``ratio · other`` (e.g. ``2/3``
        for a 3d3/2 against its 3d5/2), not fitted.
    fwhm_equal_to : str or None
        The width equals that of the named peak, not fitted.
    """

    name: str
    position_eV: float
    fwhm_eV: float = 1.0
    area: Optional[float] = None
    lineshape: Union[Lineshape, str] = "GL(30)"
    position_bounds_eV: Optional[tuple[float, float]] = None
    fwhm_bounds_eV: tuple[float, float] = (0.3, 4.0)
    area_bounds: tuple[float, float] = (0.0, math.inf)
    position_offset_from: Optional[tuple[str, float]] = None
    area_ratio_to: Optional[tuple[str, float]] = None
    fwhm_equal_to: Optional[str] = None

    def __post_init__(self) -> None:
        self.lineshape = parse_lineshape(self.lineshape)
        if self.position_bounds_eV is None:
            self.position_bounds_eV = (self.position_eV - 1.0, self.position_eV + 1.0)
        for label, bounds in (("position_bounds_eV", self.position_bounds_eV),
                              ("fwhm_bounds_eV", self.fwhm_bounds_eV), ("area_bounds", self.area_bounds)):
            if len(bounds) != 2 or not bounds[0] < bounds[1]:
                raise ValueError(f"{self.name}: {label} must be (lower, upper) with lower < upper, got {bounds}")

    @property
    def resolved_lineshape(self) -> Lineshape:
        """The lineshape as a :class:`Lineshape` (the string form is parsed on construction)."""
        return parse_lineshape(self.lineshape)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form of the specification (the lineshape as its CasaXPS string)."""
        record = dataclasses.asdict(self)
        record["lineshape"] = self.resolved_lineshape.casaxps
        for key in ("position_bounds_eV", "fwhm_bounds_eV", "area_bounds", "position_offset_from", "area_ratio_to"):
            if record[key] is not None:
                record[key] = [None if isinstance(item, float) and math.isinf(item) else item for item in record[key]]
        return record


DOUBLETS: dict[str, dict[str, Any]] = {
    "Mo 3d": {"splitting_eV": 3.13, "area_ratio": 2.0 / 3.0, "suffixes": ("5/2", "3/2")},
    "Ti 2p": {"splitting_eV": 5.7, "area_ratio": 1.0 / 2.0, "suffixes": ("3/2", "1/2")},
    "Pd 3d": {"splitting_eV": 5.26, "area_ratio": 2.0 / 3.0, "suffixes": ("5/2", "3/2")},
}
"""Nominal spin-orbit doublets, ready for ``doublet(name, position_eV, **DOUBLETS["Mo 3d"])``.

The splittings are the values quoted by the NIST X-ray Photoelectron
Spectroscopy Database (Mo 3d: 3.13 eV; Pd 3d: 5.26 eV; Ti 2p: 5.7 eV, with
5.54 eV in the metal and up to 5.7 eV in TiO2) and the Moulder et al.
*Handbook of X-ray Photoelectron Spectroscopy* (Perkin-Elmer, 1992), the
area ratios are the ``(2j + 1)`` degeneracies (``d``: 3d3/2 : 3d5/2 = 2 : 3,
``p``: 2p1/2 : 2p3/2 = 1 : 2). They are **nominal**: the measured splitting
of an oxide can differ by a tenth of an eV, so treat them as the starting
point of the constraint, not as a result.
"""


def doublet(
    name: str,
    position_eV: float,
    splitting_eV: float,
    area_ratio: float,
    fwhm_eV: float = 1.0,
    area: Optional[float] = None,
    lineshape: Union[Lineshape, str] = "GL(30)",
    suffixes: tuple[str, str] = ("5/2", "3/2"),
    *,
    position_bounds_eV: Optional[tuple[float, float]] = None,
    fwhm_bounds_eV: tuple[float, float] = (0.3, 4.0),
    equal_fwhm: bool = True,
) -> list[PeakSpec]:
    """Two :class:`PeakSpec` of a spin-orbit doublet, the second tied to the first.

    Parameters
    ----------
    name : str
        Prefix of both names, e.g. ``"Mo(VI) 3d"`` gives ``"Mo(VI) 3d5/2"``
        and ``"Mo(VI) 3d3/2"``.
    position_eV : float
        Initial centre of the first (lower binding energy, higher ``j``) peak.
    splitting_eV : float
        Fixed splitting; the second peak sits at ``position_eV + splitting_eV``.
    area_ratio : float
        Fixed ``area(second) / area(first)``, ``2/3`` for a d doublet, ``1/2`` for a p doublet.
    fwhm_eV : float, optional
        Initial width of both (default is 1.0).
    area : float or None, optional
        Initial area of the first peak; ``None`` estimates it from the data.
    lineshape : Lineshape or str, optional
        Lineshape of both (default is ``"GL(30)"``).
    suffixes : tuple of str, optional
        Appended to ``name`` (default is ``("5/2", "3/2")``).
    position_bounds_eV : tuple of float or None, optional
        Bounds of the first peak's centre (default is ±1 eV).
    fwhm_bounds_eV : tuple of float, optional
        Bounds of the width (default is 0.3 to 4 eV).
    equal_fwhm : bool, optional
        Tie the second width to the first (default is ``True``).

    Returns
    -------
    list of PeakSpec
        The two specifications, first then second.
    """
    first = PeakSpec(
        name=f"{name}{suffixes[0]}", position_eV=position_eV, fwhm_eV=fwhm_eV, area=area, lineshape=lineshape,
        position_bounds_eV=position_bounds_eV, fwhm_bounds_eV=fwhm_bounds_eV,
    )
    second = PeakSpec(
        name=f"{name}{suffixes[1]}", position_eV=position_eV + splitting_eV, fwhm_eV=fwhm_eV,
        area=None if area is None else area * area_ratio, lineshape=lineshape, fwhm_bounds_eV=fwhm_bounds_eV,
        position_offset_from=(first.name, float(splitting_eV)), area_ratio_to=(first.name, float(area_ratio)),
        fwhm_equal_to=first.name if equal_fwhm else None,
    )
    return [first, second]


# ----------------------------------------------------------------
# The fit
# ----------------------------------------------------------------


@dataclass
class FittedPeak:
    """One fitted peak: the parameters, their standard errors and the derived numbers.

    Attributes
    ----------
    name : str
        The peak's name.
    position_eV, fwhm_eV, area : float
        Fitted parameters (a constrained one is derived from its reference).
    area_fraction : float
        Share of the total fitted area.
    height : float
        Maximum of the peak curve.
    lineshape : str
        CasaXPS notation.
    position_error_eV, fwhm_error_eV, area_error : float
        Standard errors from the Jacobian, NaN when not available.
    """

    name: str
    position_eV: float
    fwhm_eV: float
    area: float
    area_fraction: float
    height: float
    lineshape: str
    position_error_eV: float = math.nan
    fwhm_error_eV: float = math.nan
    area_error: float = math.nan


@dataclass
class FitResult:
    """The outcome of :func:`fit_peaks`.

    Attributes
    ----------
    energy : numpy.ndarray
        The binding energies that were fitted.
    values : numpy.ndarray
        The signal that was fitted (background already removed).
    peaks : list of FittedPeak
        Per-peak numbers, in the order of the specifications.
    peak_curves : numpy.ndarray
        Each peak over ``energy``, shape ``(n_peaks, n_points)``.
    envelope : numpy.ndarray
        The sum of the peak curves.
    residual : numpy.ndarray
        ``values - envelope``.
    chi_square : float
        Sum of squared weighted residuals.
    reduced_chi_square : float
        ``chi_square / (n_points - n_free_parameters)``.
    n_free_parameters : int
        Parameters actually fitted (constrained ones are derived).
    success : bool
        Whether the optimiser converged.
    message : str
        The optimiser's message.
    n_function_evaluations : int
        Model evaluations the optimiser needed.
    specs : list of PeakSpec
        What was asked for.
    """

    energy: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]
    peaks: list[FittedPeak]
    peak_curves: npt.NDArray[np.float64]
    envelope: npt.NDArray[np.float64]
    residual: npt.NDArray[np.float64]
    chi_square: float
    reduced_chi_square: float
    n_free_parameters: int
    success: bool
    message: str
    n_function_evaluations: int
    specs: list[PeakSpec] = field(default_factory=list)

    @property
    def total_area(self) -> float:
        """Sum of the fitted peak areas."""
        return float(sum(peak.area for peak in self.peaks))

    def to_frame(self) -> pd.DataFrame:
        """One row per peak: name, position, FWHM, area, area fraction, height, lineshape and the errors."""
        return pd.DataFrame([dataclasses.asdict(peak) for peak in self.peaks])

    def report(self) -> str:
        """A short text table of the peaks with the fit statistics underneath."""
        lines = [f"{'peak':<22}{'position / eV':>15}{'fwhm / eV':>12}{'area':>14}{'fraction / –':>14}{'lineshape':>16}"]
        for peak in self.peaks:
            lines.append(
                f"{peak.name:<22}{peak.position_eV:>15.3f}{peak.fwhm_eV:>12.3f}{peak.area:>14.4g}"
                f"{peak.area_fraction:>14.3f}{peak.lineshape:>16}"
            )
        lines.append(
            f"reduced chi-square {self.reduced_chi_square:.4g}, {self.n_free_parameters} free parameters, "
            f"{self.energy.size} points, {'converged' if self.success else 'not converged'}: {self.message}"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary: statistics, the per-peak numbers and the specifications."""
        return {
            "success": bool(self.success),
            "message": str(self.message),
            "chi_square": float(self.chi_square),
            "reduced_chi_square": float(self.reduced_chi_square),
            "n_points": int(self.energy.size),
            "n_free_parameters": int(self.n_free_parameters),
            "n_function_evaluations": int(self.n_function_evaluations),
            "window_eV": [float(self.energy.min()), float(self.energy.max())] if self.energy.size else None,
            "peaks": [_json_safe_record(dataclasses.asdict(peak)) for peak in self.peaks],
            "specs": [spec.to_dict() for spec in self.specs],
        }


def _json_safe_record(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            value = None
        out[key] = value
    return out


@dataclass
class _Parameter:
    """Bookkeeping of one parameter of one peak: free (index into the vector) or derived."""

    kind: Literal["free", "offset", "ratio", "equal"]
    index: int = -1
    source: Optional[str] = None
    factor: float = 0.0  # delta for offset, ratio for ratio


def _build_parameter_map(specs: list[PeakSpec]) -> tuple[dict[str, dict[str, _Parameter]], list[float], list[float], list[float], list[str]]:
    """Which parameters are free, their initial values and bounds, and the derived ones."""
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Peak names must be unique, got {names}")
    table: dict[str, dict[str, _Parameter]] = {}
    initial: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    labels: list[str] = []

    def free(spec_name: str, parameter: str, value: float, bounds: tuple[float, float]) -> _Parameter:
        low, high = float(bounds[0]), float(bounds[1])
        initial.append(min(max(float(value), low), high) if math.isfinite(high) else max(float(value), low))
        lower.append(low)
        upper.append(high)
        labels.append(f"{spec_name}:{parameter}")
        return _Parameter("free", index=len(initial) - 1)

    for spec in specs:
        entry: dict[str, _Parameter] = {}
        if spec.position_offset_from is not None:
            other, delta = spec.position_offset_from
            _check_reference(spec, other, names)
            entry["position"] = _Parameter("offset", source=other, factor=float(delta))
        else:
            entry["position"] = free(spec.name, "position", spec.position_eV, spec.position_bounds_eV)  # type: ignore[arg-type]
        if spec.fwhm_equal_to is not None:
            _check_reference(spec, spec.fwhm_equal_to, names)
            entry["fwhm"] = _Parameter("equal", source=spec.fwhm_equal_to, factor=1.0)
        else:
            entry["fwhm"] = free(spec.name, "fwhm", spec.fwhm_eV, spec.fwhm_bounds_eV)
        if spec.area_ratio_to is not None:
            other, ratio = spec.area_ratio_to
            _check_reference(spec, other, names)
            entry["area"] = _Parameter("ratio", source=other, factor=float(ratio))
        else:
            entry["area"] = free(spec.name, "area", spec.area if spec.area is not None else 0.0, spec.area_bounds)
        table[spec.name] = entry
    return table, initial, lower, upper, labels


def _check_reference(spec: PeakSpec, other: str, names: list[str]) -> None:
    if other not in names:
        raise ValueError(f"Peak {spec.name!r} refers to {other!r}, which is not among {names}")
    if other == spec.name:
        raise ValueError(f"Peak {spec.name!r} cannot be constrained to itself")


def _resolve(table: dict[str, dict[str, _Parameter]], vector: npt.NDArray[np.float64],
             names: list[str]) -> dict[str, dict[str, float]]:
    """Parameter values of every peak from the free vector, following constraint chains."""
    resolved: dict[str, dict[str, float]] = {name: {} for name in names}
    pending = {(name, parameter) for name in names for parameter in ("position", "fwhm", "area")}
    while pending:
        progressed = False
        for name, parameter in sorted(pending):
            item = table[name][parameter]
            if item.kind == "free":
                resolved[name][parameter] = float(vector[item.index])
            elif parameter in resolved[item.source]:  # type: ignore[index]
                base = resolved[item.source][parameter]  # type: ignore[index]
                resolved[name][parameter] = base + item.factor if item.kind == "offset" else base * item.factor
            else:
                continue
            pending.discard((name, parameter))
            progressed = True
        if not progressed:
            raise ValueError(f"Circular constraints among {sorted(pending)}")
    return resolved


def _estimate_area(spec: PeakSpec, energy: npt.NDArray[np.float64], values: npt.NDArray[np.float64]) -> float:
    """Height at the guessed position times the width: a Gaussian-like area guess, at least a little positive."""
    if energy.size == 0:
        return 1.0
    index = int(np.argmin(np.abs(energy - spec.position_eV)))
    height = float(values[index])
    return max(height, 1e-3 * float(np.max(np.abs(values))) if values.size else 1e-3) * spec.fwhm_eV * 1.06


def _peak_height(lineshape: Lineshape, position: float, fwhm: float, area: float) -> float:
    fine = np.linspace(position - 2.0 * fwhm, position + 2.0 * fwhm, 801)
    return float(np.max(lineshape.evaluate(fine, position, fwhm, area)))


def fit_peaks(
    energy: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    peaks: list[PeakSpec],
    *,
    weights: Union[None, Literal["poisson"], npt.NDArray[np.float64]] = None,
    max_nfev: int = 2000,
) -> FitResult:
    """Fit a sum of lineshapes to a background-free signal by bounded least squares.

    Only the free parameters enter the vector that
    :func:`scipy.optimize.least_squares` (trust region reflective, bounds)
    minimises; a parameter tied to another peak by ``position_offset_from``,
    ``area_ratio_to`` or ``fwhm_equal_to`` is derived from it at every
    evaluation, chains included. Standard errors come from the Jacobian at
    the solution, ``cov = (JᵀJ)⁻¹ · reduced chi-square``, and are propagated
    onto the derived parameters.

    Parameters
    ----------
    energy : numpy.ndarray
        Binding energies in eV (any order).
    values : numpy.ndarray
        The signal after background subtraction, same length.
    peaks : list of PeakSpec
        What to fit; names must be unique.
    weights : None, "poisson" or numpy.ndarray, optional
        Residual weights. ``None`` (default) weights every point equally,
        ``"poisson"`` uses ``1 / sqrt(max(values, 1))`` (counting
        statistics, right for raw counts), an array is used as is.
    max_nfev : int, optional
        Maximum number of model evaluations (default is 2000).

    Returns
    -------
    FitResult
        The fitted peaks, the envelope, the residual and the statistics.

    Raises
    ------
    ValueError
        For duplicate names, a constraint to an unknown peak, circular
        constraints, no peaks, or arrays of different lengths.
    """
    energy = np.asarray(energy, dtype=float)
    values = np.asarray(values, dtype=float)
    if energy.ndim != 1 or energy.shape != values.shape:
        raise ValueError(f"energy {energy.shape} and values {values.shape} must be 1-D and of the same length")
    specs = [dataclasses.replace(spec) for spec in peaks]  # copies: the area guess is filled in below
    if not specs:
        raise ValueError("fit_peaks needs at least one PeakSpec")
    finite = np.isfinite(energy) & np.isfinite(values)
    energy, values = energy[finite], values[finite]
    if isinstance(weights, str):
        if weights != "poisson":
            raise ValueError(f"weights must be None, 'poisson' or an array, got {weights!r}")
        weight_array = 1.0 / np.sqrt(np.maximum(values, 1.0))
    elif weights is None:
        weight_array = np.ones_like(values)
    else:
        weight_array = np.asarray(weights, dtype=float)[finite]
        if weight_array.shape != values.shape:
            raise ValueError("weights must have the length of energy")

    for spec in specs:
        if spec.area is None and spec.area_ratio_to is None:
            spec.area = _estimate_area(spec, energy, values)
    names = [spec.name for spec in specs]
    lineshapes = {spec.name: spec.resolved_lineshape for spec in specs}
    table, initial, lower, upper, labels = _build_parameter_map(specs)
    _resolve(table, np.asarray(initial, dtype=float), names)  # raises on circular constraints before fitting

    def model(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        resolved = _resolve(table, vector, names)
        total = np.zeros_like(energy)
        for name in names:
            parameters = resolved[name]
            total += lineshapes[name].evaluate(energy, parameters["position"], parameters["fwhm"], parameters["area"])
        return total

    def residuals(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return (model(vector) - values) * weight_array

    x_scale = np.array([max(abs(value), 1e-3) for value in initial]) if initial else 1.0
    solution = least_squares(residuals, np.asarray(initial, dtype=float), bounds=(lower, upper),
                             x_scale=x_scale, max_nfev=max_nfev, method="trf")
    vector = solution.x
    resolved = _resolve(table, vector, names)

    n_points = energy.size
    n_free = vector.size
    chi_square = float(np.sum(solution.fun**2))
    degrees_of_freedom = max(n_points - n_free, 1)
    reduced = chi_square / degrees_of_freedom
    errors = _standard_errors(solution.jac, reduced, n_free)

    def error_of(name: str, parameter: str) -> float:
        item = table[name][parameter]
        if item.kind == "free":
            return float(errors[item.index])
        base = error_of(item.source, parameter)  # type: ignore[arg-type]
        return base if item.kind in ("offset", "equal") else abs(item.factor) * base

    curves = np.stack([
        lineshapes[name].evaluate(energy, resolved[name]["position"], resolved[name]["fwhm"], resolved[name]["area"])
        for name in names
    ])
    envelope = curves.sum(axis=0)
    total_area = float(sum(resolved[name]["area"] for name in names)) or math.nan
    fitted = [
        FittedPeak(
            name=name,
            position_eV=resolved[name]["position"],
            fwhm_eV=resolved[name]["fwhm"],
            area=resolved[name]["area"],
            area_fraction=resolved[name]["area"] / total_area,
            height=_peak_height(lineshapes[name], resolved[name]["position"], resolved[name]["fwhm"], resolved[name]["area"]),
            lineshape=lineshapes[name].casaxps,
            position_error_eV=error_of(name, "position"),
            fwhm_error_eV=error_of(name, "fwhm"),
            area_error=error_of(name, "area"),
        )
        for name in names
    ]
    return FitResult(
        energy=energy, values=values, peaks=fitted, peak_curves=curves, envelope=envelope,
        residual=values - envelope, chi_square=chi_square, reduced_chi_square=reduced, n_free_parameters=n_free,
        success=bool(solution.success), message=str(solution.message), n_function_evaluations=int(solution.nfev),
        specs=specs,
    )


def _standard_errors(jacobian: npt.NDArray[np.float64], reduced_chi_square: float, n_free: int) -> npt.NDArray[np.float64]:
    """Square roots of the diagonal of ``(JᵀJ)⁻¹ · reduced chi-square``; NaN where the Jacobian is degenerate."""
    errors = np.full(n_free, math.nan)
    if jacobian is None or jacobian.size == 0:
        return errors
    try:
        _, singular, v_transposed = np.linalg.svd(jacobian, full_matrices=False)
    except np.linalg.LinAlgError:
        return errors
    threshold = np.finfo(float).eps * max(jacobian.shape) * singular[0] if singular.size else 0.0
    keep = singular > threshold
    if not keep.any():
        return errors
    covariance = (v_transposed[keep].T / singular[keep] ** 2) @ v_transposed[keep] * reduced_chi_square
    diagonal = np.diag(covariance)
    errors[:] = np.sqrt(np.where(diagonal >= 0, diagonal, math.nan))
    if not keep.all():  # a degenerate direction: its errors are not meaningful
        errors[np.abs(v_transposed[~keep]).max(axis=0) > 0.5] = math.nan
    return errors
