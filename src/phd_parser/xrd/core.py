import logging
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, field_validator

from phd_parser.xrd.analysis import Lineshape, ReflectionFit, fit_reflection, scherrer

logger = logging.getLogger(__name__)

class XRDData(BaseModel):
    """Pydantic model for X-ray diffraction (XRD) data.

    Wraps an :class:`xarray.DataArray` whose primary coordinate is the
    2θ angle in degrees.  The array may be 1-D ``(angle,)`` for a
    single scan or 2-D ``(scan, angle)`` for a time-series experiment.
    All processing methods return a new ``XRDData`` instance
    (immutable API).

    Parameters
    ----------
    da : xarray.DataArray
        Intensity data with dims ``('angle',)`` or ``('scan', 'angle')``.
        The ``angle`` coordinate must be present and in degrees (2θ).
        Optional additional coords: ``'tos'`` (seconds elapsed) and
        ``'timestamp'`` (absolute datetimes).
    metadata : dict, optional
        Raw metadata extracted from the source file.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        ignored_types=(cached_property,),
    )

    # ----------------------------------------------------------------
    # Fields
    # ----------------------------------------------------------------

    # Core data: angle in degrees (2θ), consistent throughout
    da: xr.DataArray = Field(
        description=(
            "xarray DataArray with dims ('angle',) or ('scan', 'angle'). "
            "Angle coordinate is in degrees (2θ). "
            "Optional coords: 'tos' (seconds), 'timestamp' (datetime)."
        )
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw metadata extracted from the source file.",
    )

    # ----------------------------------------------------------------
    # Validators
    # ----------------------------------------------------------------

    @field_validator("da", mode="before")
    @classmethod
    def validate_da(cls, v: Any) -> xr.DataArray:
        """Check the layout of ``da``: an ``angle`` dimension, 1-D or ``(scan, angle)``; adds ``angle_units``."""
        if not isinstance(v, xr.DataArray):
            raise TypeError(f"'da' must be an xr.DataArray, got {type(v)}")
        if "angle" not in v.dims:
            raise ValueError("DataArray must have an 'angle' dimension")
        if v.ndim not in (1, 2):
            raise ValueError(f"DataArray must be 1-D or 2-D, got {v.ndim}-D")
        if v.ndim == 2 and v.dims[0] != "scan":
            raise ValueError("2-D DataArray must have dims ('scan', 'angle')")
        if "angle_units" not in v.attrs:
            v = v.assign_attrs(angle_units="deg")  # a copy: the caller's array is not edited
        return v

    # ----------------------------------------------------------------
    # Core properties
    # ----------------------------------------------------------------

    @property
    def ndim(self) -> int:
        """Number of dimensions of the underlying DataArray (1 or 2).

        Returns
        -------
        int
            ``1`` for a single scan, ``2`` for a time-series dataset.
        """
        return self.da.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying DataArray.

        Returns
        -------
        tuple of int
            Dimension sizes in the same order as :attr:`da`.dims.
        """
        return tuple(self.da.shape)

    @property
    def values(self) -> npt.NDArray:
        """Raw intensity values as a NumPy array.

        Returns
        -------
        ndarray
            Intensity data with the same shape as :attr:`shape`.
        """
        return self.da.values

    @property
    def angle(self) -> npt.NDArray:
        """2θ angle in degrees.

        Returns
        -------
        ndarray
            1-D array of 2θ values in degrees.
        """
        return self.da.coords["angle"].values


    @property
    def timestamps(self) -> Optional[pd.DatetimeIndex]:
        """Absolute timestamps per scan, or None.

        Returns
        -------
        pandas.DatetimeIndex or None
            Per-scan datetimes when the ``'timestamp'`` coordinate is
            present, otherwise ``None``.
        """
        if "timestamp" in self.da.coords:
            return pd.DatetimeIndex(self.da.coords["timestamp"].values)
        return None

    # ----------------------------------------------------------------
    # Cached unit-conversion properties
    # ----------------------------------------------------------------

    @cached_property
    def angle_rad(self) -> npt.NDArray:
        """2θ angle in radians.

        Returns
        -------
        ndarray
            1-D array of 2θ values converted to radians.
        """
        return np.deg2rad(self.angle)

    @cached_property
    def theta_deg(self) -> npt.NDArray:
        """θ (Bragg angle) in degrees.

        Returns
        -------
        ndarray
            1-D array equal to ``angle / 2``.
        """
        return self.angle / 2.0

    @cached_property
    def theta_rad(self) -> npt.NDArray:
        """θ (Bragg angle) in radians.

        Returns
        -------
        ndarray
            1-D array of Bragg angles converted to radians.
        """
        return np.deg2rad(self.theta_deg)

    def d_spacing(self, wavelength_angstrom: float = 1.5406) -> npt.NDArray:
        """Compute d-spacing in Ångströms via Bragg's law: d = λ / (2 sin θ).

        Points where sin θ = 0 (i.e. 2θ = 0°) are returned as ``nan``
        to avoid division by zero.

        Parameters
        ----------
        wavelength_angstrom : float, optional
            X-ray wavelength in Å (default is Cu Kα₁, 1.5406 Å).

        Returns
        -------
        ndarray
            d-spacing values in Ångströms, same length as :attr:`angle`.
        """
        sin_theta = np.sin(self.theta_rad)
        # Avoid division by zero at angle=0
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(sin_theta > 0, wavelength_angstrom / (2.0 * sin_theta), np.nan)
        return d

    def q_vector(self, wavelength_angstrom: float = 1.5406) -> npt.NDArray:
        """Compute the scattering vector magnitude Q in Å⁻¹: Q = 4π sin θ / λ.

        Parameters
        ----------
        wavelength_angstrom : float, optional
            X-ray wavelength in Å (default is Cu Kα₁, 1.5406 Å).

        Returns
        -------
        ndarray
            Q values in Å⁻¹, same length as :attr:`angle`.
        """
        return (4.0 * np.pi * np.sin(self.theta_rad)) / wavelength_angstrom

    # ----------------------------------------------------------------
    # Immutable transformations
    # ----------------------------------------------------------------

    def sort(self, ascending: bool = True) -> "XRDData":
        """Return a new XRDData with angles sorted.

        Parameters
        ----------
        ascending : bool, optional
            When ``True`` (default) sort from smallest to largest 2θ;
            ``False`` reverses the order.

        Returns
        -------
        XRDData
            New instance with the angle axis sorted.
        """
        da_sorted = self.da.sortby("angle", ascending=ascending)
        return XRDData(da=da_sorted, metadata=self.metadata)

    def select_angle_range(
        self,
        min_deg: Optional[float] = None,
        max_deg: Optional[float] = None,
    ) -> "XRDData":
        """Slice data to a 2θ window [min_deg, max_deg].

        Both bounds are inclusive.  Omitting a bound leaves that end
        of the range open.

        Parameters
        ----------
        min_deg : float, optional
            Lower 2θ bound in degrees.
        max_deg : float, optional
            Upper 2θ bound in degrees.

        Returns
        -------
        XRDData
            New instance restricted to the specified angular range.
        """
        mask = np.ones(self.angle.size, dtype=bool)
        if min_deg is not None:
            mask &= self.angle >= min_deg
        if max_deg is not None:
            mask &= self.angle <= max_deg

        return XRDData(da=self.da.isel(angle=mask), metadata=self.metadata)

    def smooth_savgol(
        self, window_length: int = 11, polyorder: int = 3
    ) -> "XRDData":
        """Smooth intensities with a Savitzky-Golay filter.

        For 2-D data the filter is applied independently along each
        scan row.

        Parameters
        ----------
        window_length : int, optional
            Length of the filter window in data points (default is 11).
        polyorder : int, optional
            Polynomial order used to fit within each window
            (default is 3).

        Returns
        -------
        XRDData
            New instance with smoothed intensity values.
        """
        from scipy.signal import savgol_filter

        if self.ndim == 1:
            smoothed = savgol_filter(self.values, window_length, polyorder)
        else:
            smoothed = np.apply_along_axis(
                lambda m: savgol_filter(m, window_length, polyorder),
                axis=1,
                arr=self.values,
            )
        return XRDData(da=self.da.copy(data=smoothed), metadata=self.metadata)

    def smooth_gaussian(self, sigma_deg: float = 0.05) -> "XRDData":
        """Smooth intensities with a Gaussian filter.

        The standard deviation ``sigma_deg`` is given in degrees and
        converted to index units based on the mean angular step size.
        For 2-D data the filter is applied independently along each
        scan row.

        Parameters
        ----------
        sigma_deg : float, optional
            Standard deviation of the Gaussian kernel in degrees
            (default is 0.05).

        Returns
        -------
        XRDData
            New instance with smoothed intensity values.
        """
        from scipy.ndimage import gaussian_filter1d

        # Convert sigma from degrees to index units
        step = float(np.mean(np.diff(self.angle))) if self.angle.size > 1 else 1.0
        sigma_idx = sigma_deg / step

        if self.ndim == 1:
            smoothed = gaussian_filter1d(self.values, sigma=sigma_idx)
        else:
            smoothed = np.apply_along_axis(
                lambda m: gaussian_filter1d(m, sigma=sigma_idx),
                axis=1,
                arr=self.values,
            )
        return XRDData(da=self.da.copy(data=smoothed), metadata=self.metadata)

    def smooth_moving(self, window_size: int = 5) -> "XRDData":
        """Smooth intensities with a uniform moving-average filter.

        For 2-D data the filter is applied independently along each
        scan row using ``numpy.convolve`` in ``'same'`` mode.

        Parameters
        ----------
        window_size : int, optional
            Number of data points to average (default is 5).

        Returns
        -------
        XRDData
            New instance with smoothed intensity values.

        Raises
        ------
        ValueError
            When ``window_size`` is less than 1.
        """
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        kernel = np.ones(window_size) / window_size
        if self.ndim == 1:
            smoothed = np.convolve(self.values, kernel, mode="same")
        else:
            smoothed = np.apply_along_axis(
                lambda m: np.convolve(m, kernel, mode="same"),
                axis=1,
                arr=self.values,
            )
        return XRDData(da=self.da.copy(data=smoothed), metadata=self.metadata)


    # ----------------------------------------------------------------
    # Analysis
    # ----------------------------------------------------------------

    def fit_reflection(
        self,
        two_theta_deg: float,
        window_deg: float = 1.5,
        lineshape: Lineshape = "pseudo_voigt",
        *,
        scan: Optional[int] = None,
    ) -> ReflectionFit:
        """Fit one reflection with a pseudo-Voigt plus a linear background.

        A thin wrapper of :func:`phd_parser.xrd.analysis.fit_reflection`
        over this pattern; the data is not changed.

        Parameters
        ----------
        two_theta_deg : float
            Expected position of the reflection in degrees, the window centre.
        window_deg : float, optional
            Full width of the fit window in degrees (default is 1.5).
        lineshape : {"pseudo_voigt", "gaussian", "lorentzian"}, optional
            The line to fit (default is ``"pseudo_voigt"``).
        scan : int, optional
            For 2-D data, the index of the scan to fit. Required then,
            ignored for a single pattern.

        Returns
        -------
        ReflectionFit
            ``position_deg``, ``fwhm_deg``, ``height``, ``area``, ``eta``
            and ``background`` of the fit, plus ``model(angle_deg)`` to
            draw it.

        Raises
        ------
        ValueError
            For 2-D data without ``scan``, an unknown lineshape, or a
            window holding too few points.
        """
        if self.ndim == 2:
            if scan is None:
                raise ValueError("2-D data needs the index of the scan to fit (scan=...)")
            values = self.da.isel(scan=scan).values
        else:
            values = self.values
        return fit_reflection(self.angle, values, two_theta_deg, window_deg=window_deg, lineshape=lineshape)

    def scherrer_size(
        self,
        two_theta_deg: float,
        fwhm_deg: float,
        *,
        shape_factor: float = 0.9,
        wavelength_angstrom: Optional[float] = None,
        instrumental_fwhm_deg: float = 0.0,
    ) -> float:
        """Scherrer crystallite size in nm from a fitted reflection width.

        Parameters
        ----------
        two_theta_deg : float
            Position of the reflection, 2θ in degrees.
        fwhm_deg : float
            Observed full width at half maximum in degrees 2θ, e.g.
            ``fit_reflection(...).fwhm_deg``.
        shape_factor : float, optional
            The Scherrer constant ``K`` (default is 0.9).
        wavelength_angstrom : float, optional
            X-ray wavelength in Å. Default is ``da.attrs["wavelength_angstrom"]``
            when the reader set it (Bruker: from the anode), else Cu Kα1,
            1.5406 Å.
        instrumental_fwhm_deg : float, optional
            Instrumental broadening in degrees 2θ, subtracted in
            quadrature (default is 0, no correction).

        Returns
        -------
        float
            Crystallite size in nm.
        """
        if wavelength_angstrom is None:
            wavelength_angstrom = self.da.attrs.get("wavelength_angstrom") or 1.5406
        return scherrer(
            two_theta_deg,
            fwhm_deg,
            wavelength_angstrom=float(wavelength_angstrom),
            shape_factor=shape_factor,
            instrumental_fwhm_deg=instrumental_fwhm_deg,
        )

    # ----------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------

    @classmethod
    def from_bruker_xy(
        cls,
        filepath: Union[str, Path],
        normalize: bool = False,
    ) -> "XRDData":
        """Read a Bruker DIFFRAC.EVA ``.xy`` export (D8 Advance-ECO).

        The header line (operator, anode, scan type, time per step) goes
        into :attr:`metadata`, see :func:`phd_parser.xrd.bruker.read_xy`;
        ``wavelength_angstrom`` (Kα1 of the anode), ``source_file`` and
        ``anode`` go into ``da.attrs`` as well, where ``scherrer_size``
        finds the wavelength.

        Parameters
        ----------
        filepath : str or Path
            Path to the ``.xy`` file.
        normalize : bool, optional
            When ``True``, divide the intensity by its maximum
            (default is ``False``).

        Returns
        -------
        XRDData
            The pattern, angle ascending.
        """
        from phd_parser.xrd.bruker import read_xy

        parsed = read_xy(filepath)
        angle = parsed["data"]["angle"]
        values = parsed["data"]["values"]
        order = np.argsort(angle)
        angle, values = angle[order], values[order]
        if normalize:
            values = values / values.max()

        meta = parsed["meta"]
        da = cls._build_da(angle=angle, values=values)
        da.attrs.update(
            {key: meta[key] for key in ("wavelength_angstrom", "source_file", "anode") if meta.get(key) is not None}
        )
        return cls(da=da, metadata=meta)

    @classmethod
    def from_e1290(
        cls,
        filepath: Union[str, Path],
        normalize: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "XRDData":
        """Read a Bruker/E1290-style two-column XY file.

        Parameters
        ----------
        filepath : str or Path
            Path to the ``.xy`` file to parse.
        normalize : bool, optional
            When ``True``, divide intensity by its maximum on load
            (default is ``False``).
        metadata : dict, optional
            Additional metadata to attach to the returned instance.

        Returns
        -------
        XRDData
            Parsed dataset with angle in degrees and intensity values.
        """
        from phd_parser.xrd.xrd_e1290 import read_xy_e1290

        filepath = Path(filepath)
        data = read_xy_e1290(filepath, normalize=normalize)

        return cls(
            da=cls._build_da(
                angle=data[0],
                values=data[1],
            ),
            metadata=metadata or {},
        )

    # ----------------------------------------------------------------
    # Dunder helpers
    # ----------------------------------------------------------------

    def __repr__(self) -> str:
        angle = self.angle
        angle_range = (
            f"{angle.min():.3f}–{angle.max():.3f} °"
            if angle.size
            else "empty"
        )
        dims = dict(zip(self.da.dims, self.da.shape))
        return (
            f"XRDData("
            f"shape={dims}, "
            f"2θ={angle_range}"
            f")"
        )

    def __len__(self) -> int:
        return self.da.sizes.get("scan", 1)

    # ----------------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _build_da(
        angle: npt.NDArray,
        values: npt.NDArray,
    ) -> xr.DataArray:
        """Build the canonical DataArray from validated raw arrays."""
        coords: dict[str, Any] = {"angle": angle}
        dims: list[str]

        if values.ndim == 1:
            dims = ["angle"]
        else:
            dims = ["scan", "angle"]
            coords["scan"] = np.arange(values.shape[0])

        return xr.DataArray(
            data=values,
            coords=coords,
            dims=dims,
            attrs={
                "angle_units": "deg",
            }
        )
