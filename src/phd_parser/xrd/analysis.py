"""Single-reflection fitting and Scherrer crystallite size for XRD patterns.

Pure functions over plain arrays; :class:`phd_parser.xrd.XRDData` wraps
them as ``fit_reflection`` and ``scherrer_size``.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import numpy.typing as npt

from phd_parser.kernels import gaussian, lorentzian, pseudo_voigt, pseudo_voigt_area

logger = logging.getLogger(__name__)

Lineshape = Literal["pseudo_voigt", "gaussian", "lorentzian"]


@dataclass
class ReflectionFit:
    """Result of :func:`fit_reflection`: one line plus a linear background.

    The background is ``background[0] + background[1] * (x - position_deg)``,
    so ``background[0]`` is the background level under the maximum.
    """

    position_deg: float
    fwhm_deg: float
    height: float
    area: float
    eta: float
    background: tuple[float, float]
    lineshape: str
    window_deg: tuple[float, float]
    n_points: int
    standard_errors: dict[str, float] = field(default_factory=dict, repr=False)

    def model(self, angle_deg: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Fitted line plus background at the given angles.

        Parameters
        ----------
        angle_deg : ndarray of float
            2θ in degrees.

        Returns
        -------
        ndarray of float
            The fitted curve.
        """
        angle_deg = np.asarray(angle_deg, dtype=np.float64)
        return self.line(angle_deg) + self.background_at(angle_deg)

    def line(self, angle_deg: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Fitted line without the background.

        Parameters
        ----------
        angle_deg : ndarray of float
            2θ in degrees.

        Returns
        -------
        ndarray of float
            The line alone.
        """
        return pseudo_voigt(np.asarray(angle_deg, dtype=np.float64), self.position_deg, self.fwhm_deg, self.height, self.eta)

    def background_at(self, angle_deg: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Fitted linear background at the given angles.

        Parameters
        ----------
        angle_deg : ndarray of float
            2θ in degrees.

        Returns
        -------
        ndarray of float
            The background alone.
        """
        angle_deg = np.asarray(angle_deg, dtype=np.float64)
        return self.background[0] + self.background[1] * (angle_deg - self.position_deg)

    def to_dict(self) -> dict[str, Any]:
        """The fit as a plain dict (JSON-safe).

        Returns
        -------
        dict
            Every field, tuples as lists.
        """
        return {
            "position_deg": self.position_deg,
            "fwhm_deg": self.fwhm_deg,
            "height": self.height,
            "area": self.area,
            "eta": self.eta,
            "background": list(self.background),
            "lineshape": self.lineshape,
            "window_deg": list(self.window_deg),
            "n_points": self.n_points,
            "standard_errors": dict(self.standard_errors),
        }


def _estimate_fwhm(x: npt.NDArray[np.float64], signal: npt.NDArray[np.float64], peak_index: int) -> float:
    """Width of ``signal`` at half of its value at ``peak_index``."""
    half = 0.5 * signal[peak_index]
    left = peak_index
    while left > 0 and signal[left] > half:
        left -= 1
    right = peak_index
    while right < signal.size - 1 and signal[right] > half:
        right += 1
    return float(max(x[right] - x[left], 2.0 * float(np.median(np.diff(x)))))


def fit_reflection(
    angle_deg: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    two_theta_deg: float,
    window_deg: float = 1.5,
    lineshape: Lineshape = "pseudo_voigt",
) -> ReflectionFit:
    """Fit one reflection with a line plus a linear background.

    Fits inside ``[two_theta_deg - window_deg / 2, two_theta_deg + window_deg / 2]``
    with :func:`scipy.optimize.curve_fit`. The line is a pseudo-Voigt
    with a free Lorentzian fraction, or a pure Gaussian or Lorentzian
    (fraction fixed to 0 or 1) when ``lineshape`` says so.

    Parameters
    ----------
    angle_deg : ndarray of float
        2θ in degrees, any order.
    values : ndarray of float
        Intensities, same length as ``angle_deg``.
    two_theta_deg : float
        Expected position of the reflection in degrees, the window centre.
    window_deg : float, optional
        Full width of the fit window in degrees (default is 1.5).
    lineshape : {"pseudo_voigt", "gaussian", "lorentzian"}, optional
        The line to fit (default is ``"pseudo_voigt"``).

    Returns
    -------
    ReflectionFit
        Position, full width at half maximum, height, area, Lorentzian
        fraction and background of the fit, with one standard error per
        fitted parameter.

    Raises
    ------
    ValueError
        For an unknown ``lineshape``, or a window holding fewer than
        seven points.
    RuntimeError
        When the fit does not converge.
    """
    from scipy.optimize import curve_fit

    if lineshape not in ("pseudo_voigt", "gaussian", "lorentzian"):
        raise ValueError(f"lineshape must be 'pseudo_voigt', 'gaussian' or 'lorentzian', got {lineshape!r}")

    angle_deg = np.asarray(angle_deg, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(angle_deg)
    angle_deg, values = angle_deg[order], values[order]

    lower, upper = two_theta_deg - window_deg / 2.0, two_theta_deg + window_deg / 2.0
    mask = (angle_deg >= lower) & (angle_deg <= upper) & np.isfinite(values)
    x, y = angle_deg[mask], values[mask]
    if x.size < 7:
        raise ValueError(
            f"only {x.size} points in the window {lower:.3f}–{upper:.3f} °, need at least 7"
        )

    # starting values: background through the window ends, peak at the maximum above it
    edge = max(3, x.size // 10)
    background_left, background_right = float(np.median(y[:edge])), float(np.median(y[-edge:]))
    slope = (background_right - background_left) / (x[-1] - x[0])
    baseline = background_left + slope * (x - x[0])
    above = y - baseline
    peak_index = int(np.argmax(above))
    center0 = float(x[peak_index])
    height0 = max(float(above[peak_index]), 1e-12)
    fwhm0 = min(_estimate_fwhm(x, above, peak_index), window_deg / 2.0)
    step = float(np.median(np.diff(x)))

    def background_model(x_, intercept, slope_):
        return intercept + slope_ * (x_ - center0)

    if lineshape == "pseudo_voigt":

        def model(x_, center, fwhm, height, eta, intercept, slope_):
            return pseudo_voigt(x_, center, fwhm, height, eta) + background_model(x_, intercept, slope_)

        p0 = [center0, fwhm0, height0, 0.5, float(baseline[peak_index]), slope]
        bounds = (
            [lower, step, 0.0, 0.0, -np.inf, -np.inf],
            [upper, window_deg, np.inf, 1.0, np.inf, np.inf],
        )
        names = ["position_deg", "fwhm_deg", "height", "eta", "background_level", "background_slope"]
    else:
        eta_fixed = 0.0 if lineshape == "gaussian" else 1.0

        def model(x_, center, fwhm, height, intercept, slope_):
            return pseudo_voigt(x_, center, fwhm, height, eta_fixed) + background_model(x_, intercept, slope_)

        p0 = [center0, fwhm0, height0, float(baseline[peak_index]), slope]
        bounds = (
            [lower, step, 0.0, -np.inf, -np.inf],
            [upper, window_deg, np.inf, np.inf, np.inf],
        )
        names = ["position_deg", "fwhm_deg", "height", "background_level", "background_slope"]

    try:
        popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=20000)
    except RuntimeError as error:
        raise RuntimeError(f"fit of the reflection at {two_theta_deg} ° did not converge: {error}") from error

    with np.errstate(invalid="ignore"):
        errors = np.sqrt(np.diag(pcov))
    standard_errors = {name: float(error) for name, error in zip(names, errors)}

    if lineshape == "pseudo_voigt":
        center, fwhm, height, eta, intercept, slope_fit = (float(value) for value in popt)
    else:
        center, fwhm, height, intercept, slope_fit = (float(value) for value in popt)
        eta = eta_fixed

    # the background is reported relative to the fitted position, not the start value
    level_at_position = intercept + slope_fit * (center - center0)

    if abs(center - center0) > window_deg / 4.0:
        logger.warning(
            "reflection fitted at %.3f °, %.3f ° away from the maximum at %.3f °",
            center, abs(center - center0), center0,
        )
    if fwhm <= 1.5 * step:
        logger.warning(
            "width of the reflection at %.3f ° is at the lower bound (%.4f °, one step): "
            "a single point, not a reflection",
            center, fwhm,
        )
    elif fwhm >= 0.95 * window_deg:
        logger.warning(
            "width of the reflection at %.3f ° (%.3f °) fills the window of %.3f °: widen the window",
            center, fwhm, window_deg,
        )

    return ReflectionFit(
        position_deg=center,
        fwhm_deg=fwhm,
        height=height,
        area=pseudo_voigt_area(fwhm, height, eta),
        eta=eta,
        background=(float(level_at_position), slope_fit),
        lineshape=lineshape,
        window_deg=(float(lower), float(upper)),
        n_points=int(x.size),
        standard_errors=standard_errors,
    )


def scherrer(
    two_theta_deg: float,
    fwhm_deg: float,
    wavelength_angstrom: float = 1.5406,
    shape_factor: float = 0.9,
    instrumental_fwhm_deg: float = 0.0,
) -> float:
    """Scherrer crystallite size in nm from the width of one reflection.

    ``size = K * λ / (β cos θ)``, with ``β`` the full width at half
    maximum in radians after the instrumental width is subtracted in
    quadrature: ``β = sqrt(fwhm² - instrumental_fwhm²)``.

    Parameters
    ----------
    two_theta_deg : float
        Position of the reflection, 2θ in degrees.
    fwhm_deg : float
        Observed full width at half maximum in degrees 2θ.
    wavelength_angstrom : float, optional
        X-ray wavelength in Å (default is Cu Kα1, 1.5406 Å).
    shape_factor : float, optional
        The Scherrer constant ``K`` (default is 0.9).
    instrumental_fwhm_deg : float, optional
        Instrumental broadening in degrees 2θ, subtracted in quadrature
        (default is 0, no correction).

    Returns
    -------
    float
        Crystallite size in nm.

    Raises
    ------
    ValueError
        When the instrumental width is not smaller than the observed one,
        or either width is not positive.
    """
    if fwhm_deg <= 0.0:
        raise ValueError(f"fwhm_deg must be positive, got {fwhm_deg}")
    if instrumental_fwhm_deg < 0.0:
        raise ValueError(f"instrumental_fwhm_deg must not be negative, got {instrumental_fwhm_deg}")
    if instrumental_fwhm_deg >= fwhm_deg:
        raise ValueError(
            f"instrumental width {instrumental_fwhm_deg} ° is not below the observed width {fwhm_deg} °, "
            "the sample broadening is not resolved"
        )
    beta_rad = np.deg2rad(np.sqrt(fwhm_deg**2 - instrumental_fwhm_deg**2))
    theta_rad = np.deg2rad(two_theta_deg / 2.0)
    size_angstrom = shape_factor * wavelength_angstrom / (beta_rad * np.cos(theta_rad))
    return float(size_angstrom / 10.0)
