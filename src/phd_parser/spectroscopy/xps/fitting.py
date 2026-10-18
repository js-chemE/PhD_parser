"""CasaXPS-style peak fitting of one XPS region: peak specifications, constraints and the fit.

The module replaces the curve-fitting step of CasaXPS so an analysis can be
done in Python end to end: a :class:`PeakSpec` describes one peak with its
initial guesses, bounds and the CasaXPS-style constraints to another peak
(spin-orbit splitting, area ratio and width ratio, each fixed or in a
range), :func:`doublet` builds a spin-orbit pair with the nominal splittings
of :data:`DOUBLETS`, and :func:`fit_peaks` is a bounded least-squares fit that
returns a :class:`FitResult` (with AIC and BIC for model selection).
:meth:`XPSData.fit_peaks <phd_parser.spectroscopy.xps.core.XPSData.fit_peaks>`
wraps all of it for a region.

The lineshapes themselves (``GL(30)``, ``SGL(30)``, ``LA(1.53,243)``) are
plain kernels and live in :mod:`phd_parser.kernels`; a peak's ``lineshape``
is a :class:`~phd_parser.kernels.Lineshape` or its CasaXPS string. Areas are
integrated areas in ``data_type`` units times eV, positions the centre
parameter CasaXPS reports, widths the true full width at half maximum.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import least_squares

from phd_parser.kernels import Lineshape, parse_lineshape


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
    position_offset_from : tuple of (str, float) or (str, (float, float)) or None
        ``(other_name, delta_eV)``: the centre is ``other + delta_eV`` (a
        fixed spin-orbit splitting or chemical shift). With a range,
        ``(other_name, (low, high))``, the offset itself is fitted inside
        it (CasaXPS's ``#`` constraint), e.g. a chemical shift known only
        to within a few tenths of an eV.
    area_ratio_to : tuple of (str, float) or (str, (float, float)) or None
        ``(other_name, ratio)``: the area is ``ratio · other`` (e.g. ``2/3``
        for a 3d3/2 against its 3d5/2); a range lets the ratio float.
    fwhm_equal_to : str or None
        The width equals that of the named peak, not fitted.
    fwhm_ratio_to : tuple of (str, float) or (str, (float, float)) or None
        ``(other_name, ratio)``: the width is ``ratio · other``; with a range
        the ratio is fitted. For a 2p1/2 line, broadened by the Coster-Kronig
        decay, ``("Ti(IV) 2p3/2", (1.2, 1.8))``. Not together with
        `fwhm_equal_to`.
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
    fwhm_ratio_to: Optional[tuple[str, Any]] = None

    def __post_init__(self) -> None:
        self.lineshape = parse_lineshape(self.lineshape)
        if self.fwhm_equal_to is not None and self.fwhm_ratio_to is not None:
            raise ValueError(f"{self.name}: give fwhm_equal_to or fwhm_ratio_to, not both")
        for label, relation in (("position_offset_from", self.position_offset_from),
                                ("area_ratio_to", self.area_ratio_to), ("fwhm_ratio_to", self.fwhm_ratio_to)):
            if relation is not None:
                value = relation[1]
                if isinstance(value, (tuple, list)) and not (len(value) == 2 and value[0] < value[1]):
                    raise ValueError(f"{self.name}: the range of {label} must be (low, high) with low < high, got {value}")
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
        for key in ("position_bounds_eV", "fwhm_bounds_eV", "area_bounds"):
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
    def aic(self) -> float:
        """Akaike information criterion of the fit, ``n ln(chi^2 / n) + 2k``; lower is better."""
        n = self.energy.size
        return float(n * math.log(max(self.chi_square, 1e-300) / n) + 2 * self.n_free_parameters)

    @property
    def bic(self) -> float:
        """Bayesian information criterion, ``n ln(chi^2 / n) + k ln n``; a model with one more
        doublet earns its place only when the BIC drops (by about 10 or more for strong evidence)."""
        n = self.energy.size
        return float(n * math.log(max(self.chi_square, 1e-300) / n) + self.n_free_parameters * math.log(n))

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
            "aic": self.aic,
            "bic": self.bic,
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
    factor: float = 0.0  # delta for offset, ratio for ratio; ignored when relation_index is set
    relation_index: int = -1  # index of a fitted offset or ratio (a ranged relation), -1 when fixed


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

    def related(spec: PeakSpec, parameter: str, kind: str, relation: tuple[str, Any]) -> _Parameter:
        other, value = relation
        _check_reference(spec, other, names)
        if isinstance(value, (tuple, list)):        # ranged: the offset or ratio is fitted
            low, high = float(value[0]), float(value[1])
            item = free(spec.name, f"{parameter} {kind} to {other}", 0.5 * (low + high), (low, high))
            return _Parameter(kind, source=other, relation_index=item.index)  # type: ignore[arg-type]
        return _Parameter(kind, source=other, factor=float(value))  # type: ignore[arg-type]

    for spec in specs:
        entry: dict[str, _Parameter] = {}
        if spec.position_offset_from is not None:
            entry["position"] = related(spec, "position", "offset", spec.position_offset_from)
        else:
            entry["position"] = free(spec.name, "position", spec.position_eV, spec.position_bounds_eV)  # type: ignore[arg-type]
        if spec.fwhm_equal_to is not None:
            _check_reference(spec, spec.fwhm_equal_to, names)
            entry["fwhm"] = _Parameter("equal", source=spec.fwhm_equal_to, factor=1.0)
        elif spec.fwhm_ratio_to is not None:
            entry["fwhm"] = related(spec, "fwhm", "ratio", spec.fwhm_ratio_to)
        else:
            entry["fwhm"] = free(spec.name, "fwhm", spec.fwhm_eV, spec.fwhm_bounds_eV)
        if spec.area_ratio_to is not None:
            entry["area"] = related(spec, "area", "ratio", spec.area_ratio_to)
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
                factor = float(vector[item.relation_index]) if item.relation_index >= 0 else item.factor
                resolved[name][parameter] = base + factor if item.kind == "offset" else base * factor
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
        if spec.fwhm_ratio_to is not None and isinstance(spec.fwhm_ratio_to[1], (tuple, list)):
            spec.fwhm_ratio_to = (spec.fwhm_ratio_to[0], tuple(spec.fwhm_ratio_to[1]))
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
        if item.relation_index >= 0:   # ranged relation: add the relation's own error in quadrature
            relation_error = float(errors[item.relation_index])
            if item.kind == "offset":
                return math.hypot(base, relation_error)
            factor = float(vector[item.relation_index])
            return math.hypot(abs(factor) * base, abs(resolved[item.source][parameter]) * relation_error)  # type: ignore[index]
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
