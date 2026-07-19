import logging
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scipy import constants as const

from phd_parser.infrared import omnic

logger = logging.getLogger(__name__)

XLabel = Literal["wavenumber", "frequency", "energy"]
IRDataType = Literal["single_beam", "absorbance", "transmittance", "reflectance", "kubelka_munk", "log_1_r"]

# Maps omnic header 'title' (vlabel) → IRData data_type attribute.
# Keys are the strings produced by omnic._read_header ymap.
_OMNIC_VLABEL_TO_DATA_TYPE: dict[str, IRDataType] = {
    "absorbance": "absorbance",
    "transmittance": "transmittance",
    "reflectance": "reflectance",
    "single beam": "single_beam",
    "Kubelka_Munk": "kubelka_munk",
    "log(1/R)": "log_1_r",
}


class IRData(BaseModel):
    """Immutable wrapper around an xarray Dataset for infrared spectroscopy data.

    Spectra are stored in SI units (wavenumber in m⁻¹). All processing methods
    return a new ``IRData`` instance rather than mutating ``self``.

    Attributes
    ----------
    ds : xr.Dataset
        Dataset containing a ``"data"`` variable with dims ``('wavenumber',)``
        for a single spectrum or ``('scan', 'wavenumber')`` for a time series.
        Wavenumber coordinate is in m⁻¹. An optional ``'tos'`` coordinate
        (elapsed seconds, attached to the ``scan`` dim), a ``'background'``
        variable (1-D, single_beam), and a ``'baseline'`` variable (same dims
        as ``'data'``) may also be present.  Dataset-level attrs
        hold ``data_type``, ``wavenumber_unit``, ``tos_start`` (ISO string),
        and provenance keys written by processing methods.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        ignored_types=(cached_property,),
    )

    # ----------------------------------------------------------------
    # Fields
    # ----------------------------------------------------------------

    # Core data — SI units (m⁻¹). ds["data"] holds the spectrum. Coords: 'wavenumber' always,
    # 'scan'+'tos' for 2-D. All attrs are at dataset level (ds.attrs).
    # 'timestamp' is not stored — derived on demand from tos + ds.attrs['tos_start'].
    ds: xr.Dataset = Field(
        description=(
            "xarray Dataset with a 'data' variable with dims ('wavenumber',) or ('scan', 'wavenumber'). "
            "Wavenumber in m⁻¹. Optional coord: 'tos' (seconds). Attrs at dataset level."
        )
    )

    # ----------------------------------------------------------------
    # Validators
    # ----------------------------------------------------------------

    @field_validator("ds", mode="before")
    @classmethod
    def validate_ds(cls, v: Any) -> xr.Dataset:
        if not isinstance(v, xr.Dataset):
            raise TypeError(f"'ds' must be an xr.Dataset, got {type(v)}")
        if "data" not in v:
            raise ValueError("Dataset must contain a 'data' variable")
        data_var = v["data"]
        if "wavenumber" not in data_var.dims:
            raise ValueError("Dataset 'data' variable must have a 'wavenumber' dimension")
        if data_var.ndim not in (1, 2):
            raise ValueError(f"Dataset 'data' variable must be 1-D or 2-D, got {data_var.ndim}-D")
        if data_var.ndim == 2 and data_var.dims[0] != "scan":
            raise ValueError("2-D Dataset 'data' variable must have dims ('scan', 'wavenumber')")
        if "background" in v:
            cls._validate_background_var(v)
        if "baseline" in v:
            cls._validate_baseline_var(v)
        return v

    @classmethod
    def _validate_background_var(cls, ds: xr.Dataset) -> None:
        bg = ds["background"]
        if "wavenumber" not in bg.dims:
            raise ValueError("Background variable must have a 'wavenumber' dimension")
        if bg.ndim != 1:
            raise ValueError(f"Background variable must be 1-D, got {bg.ndim}-D")
        if not np.allclose(bg.coords["wavenumber"].values, ds["data"].coords["wavenumber"].values):
            raise ValueError("Background wavenumber axis does not match data wavenumber axis")

    @classmethod
    def _validate_baseline_var(cls, ds: xr.Dataset) -> None:
        bl = ds["baseline"]
        data = ds["data"]
        if bl.dims != data.dims:
            raise ValueError(f"Baseline variable dims {bl.dims} must match data dims {data.dims}")
        if bl.shape != data.shape:
            raise ValueError(f"Baseline variable shape {bl.shape} must match data shape {data.shape}")

    @model_validator(mode="after")
    def validate_attrs(self) -> "IRData":
        return self

    # ----------------------------------------------------------------
    # Core properties
    # ----------------------------------------------------------------

    @property
    def ndim(self) -> int:
        """Number of dimensions of the data variable (1 or 2).

        Returns
        -------
        int
            ``1`` for a single spectrum, ``2`` for a time series
            ``(scan, wavenumber)``.
        """
        return self.ds["data"].ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the data variable.

        Returns
        -------
        tuple of int
            ``(n_wavenumber,)`` for 1-D data or
            ``(n_scan, n_wavenumber)`` for 2-D data.
        """
        return tuple(self.ds["data"].shape)

    @property
    def values(self) -> npt.NDArray:
        """Raw spectral intensity values as a NumPy array.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_wavenumber,)`` or ``(n_scan, n_wavenumber)``.
        """
        return self.ds["data"].values

    @property
    def wavenumber(self) -> npt.NDArray:
        """Wavenumber axis in SI units (m⁻¹).

        Returns
        -------
        numpy.ndarray
            1-D array of wavenumber values in m⁻¹.
        """
        # SI units (m⁻¹)
        return self.ds.coords["wavenumber"].values

    @property
    def wavenumber_per_cm(self) -> npt.NDArray:
        """Wavenumber axis in cm⁻¹.

        Returns
        -------
        numpy.ndarray
            1-D array of wavenumber values in cm⁻¹.
        """
        return self.wavenumber / 100.0

    @property
    def tos(self) -> Optional[npt.NDArray]:
        """Elapsed time of each scan in seconds since ``tos_start``.

        Returns
        -------
        numpy.ndarray or None
            1-D array of elapsed seconds, or ``None`` if the ``'tos'``
            coordinate is absent.
        """
        # Elapsed seconds since first scan
        if "tos" in self.ds.coords:
            return self.ds.coords["tos"].values
        return None

    @property
    def tos_start(self) -> Optional[pd.Timestamp]:
        """Absolute start time of the measurement.

        Returns
        -------
        pandas.Timestamp or None
            Parsed from ``ds.attrs['tos_start']``, or ``None`` if not set.
        """
        # Parse from attributes; not stored as a coordinate since it's a single value applying to all scans
        raw = self.ds.attrs.get("tos_start")
        if raw is None:
            return None
        return pd.Timestamp(raw)

    @property
    def data_type(self) -> Optional[IRDataType]:
        """Spectral quantity stored in ``ds['data']``.

        Returns
        -------
        str or None
            One of ``'single_beam'``, ``'absorbance'``, ``'transmittance'``,
            ``'reflectance'``, ``'log_1_r'``, ``'kubelka_munk'``, or ``None``
            if not set.
        """
        return self.ds.attrs.get("data_type")

    @property
    def has_background(self) -> bool:
        """Whether a background spectrum is stored in the dataset.

        Returns
        -------
        bool
            ``True`` if ``ds['background']`` exists.
        """
        return "background" in self.ds

    @property
    def background(self) -> Optional[npt.NDArray]:
        """Background spectrum values (single_beam, 1-D).

        Returns
        -------
        numpy.ndarray or None
            1-D array of background intensities, or ``None`` if no background
            has been assigned.
        """
        if "background" not in self.ds:
            return None
        return self.ds["background"].values

    @property
    def background_data_type(self) -> Optional[IRDataType]:
        """Data type label of the stored background variable.

        Returns
        -------
        str or None
            Value of ``ds['background'].attrs['data_type']``, or ``None`` if
            no background is present.
        """
        if "background" not in self.ds:
            return None
        return self.ds["background"].attrs.get("data_type")

    @property
    def has_baseline(self) -> bool:
        """Whether a baseline curve is stored in the dataset.

        Returns
        -------
        bool
            ``True`` if ``ds['baseline']`` exists.
        """
        return "baseline" in self.ds

    @property
    def baseline(self) -> Optional[npt.NDArray]:
        """Stored baseline curve, in the units of the current ``data_type``.

        Returns
        -------
        numpy.ndarray or None
            Array with the same shape as ``values``, or ``None`` if no
            baseline has been subtracted.
        """
        if "baseline" not in self.ds:
            return None
        return self.ds["baseline"].values

    @property
    def data_unbaselined(self) -> npt.NDArray:
        """Spectral values before baseline subtraction (``data + baseline``).

        Returns
        -------
        numpy.ndarray
            Array with the same shape as ``values``.  Equal to ``values``
            unchanged if no baseline has been subtracted.
        """
        if not self.has_baseline:
            return self.values.copy()
        return self.values + self.baseline

    @property
    def timestamps(self) -> Optional[pd.DatetimeIndex]:
        """Absolute datetime of each scan derived from ``tos`` and ``tos_start``.

        Returns
        -------
        pandas.DatetimeIndex or None
            Index of absolute timestamps, or ``None`` if either ``tos`` or
            ``tos_start`` is missing.
        """
        # Derived from tos + tos_start; not stored as a coordinate
        if self.tos is None or self.tos_start is None:
            return None
        return pd.DatetimeIndex([
            self.tos_start + pd.Timedelta(seconds=float(t)) for t in self.tos
        ])

    # ----------------------------------------------------------------
    # Cached unit-conversion properties
    # ----------------------------------------------------------------

    @cached_property
    def wavelength(self) -> npt.NDArray:
        """Wavelength axis in metres (``1 / wavenumber``).

        Returns
        -------
        numpy.ndarray
            1-D array of wavelength values in m.
        """
        # metres
        return 1.0 / self.wavenumber

    @cached_property
    def wavelength_nm(self) -> npt.NDArray:
        """Wavelength axis in nanometres.

        Returns
        -------
        numpy.ndarray
            1-D array of wavelength values in nm.
        """
        return self.wavelength * 1e9

    @cached_property
    def wavelength_mum(self) -> npt.NDArray:
        """Wavelength axis in micrometres.

        Returns
        -------
        numpy.ndarray
            1-D array of wavelength values in µm.
        """
        return self.wavelength * 1e6

    @cached_property
    def frequency(self) -> npt.NDArray:
        """Frequency axis in hertz (``wavenumber * c``).

        Returns
        -------
        numpy.ndarray
            1-D array of frequency values in Hz.
        """
        # Hz
        return self.wavenumber * const.c

    @cached_property
    def energy(self) -> npt.NDArray:
        """Photon energy axis in joules (``wavenumber * h * c``).

        Returns
        -------
        numpy.ndarray
            1-D array of photon energies in J.
        """
        # Joules
        return self.wavenumber * const.Planck * const.c

    @cached_property
    def energy_eV(self) -> npt.NDArray:
        """Photon energy axis in electronvolts.

        Returns
        -------
        numpy.ndarray
            1-D array of photon energies in eV.
        """
        return self.energy / const.electron_volt

    @cached_property
    def energy_kJ_per_mol(self) -> npt.NDArray:
        """Molar photon energy axis in kJ mol⁻¹.

        Returns
        -------
        numpy.ndarray
            1-D array of molar photon energies in kJ mol⁻¹.
        """
        return 1e-3 * self.energy * const.Avogadro # kJ/mol

    # ----------------------------------------------------------------
    # Get
    # ----------------------------------------------------------------

    def get_scan(self, scan_index: int) -> npt.NDArray:
        """Return the intensity values of a single scan by integer index.

        Parameters
        ----------
        scan_index : int
            Zero-based index of the scan to retrieve.

        Returns
        -------
        numpy.ndarray
            1-D array of shape ``(n_wavenumber,)``.

        Raises
        ------
        ValueError
            If the data are 1-D (no scan dimension).
        IndexError
            If ``scan_index`` is outside ``[0, n_scan)``.
        """
        if self.ndim == 1:
            raise ValueError("get_scan requires 2-D data")
        if not (0 <= scan_index < self.shape[0]):
            raise IndexError(
                f"scan_index {scan_index} out of bounds for {self.shape[0]} scans"
            )
        return self.ds["data"].isel(scan=scan_index).values

    def get_baseline_scan(self, scan_index: int) -> npt.NDArray:
        """Return the stored baseline curve of a single scan by integer index.

        Parameters
        ----------
        scan_index : int
            Zero-based index of the scan to retrieve.

        Returns
        -------
        numpy.ndarray
            1-D array of shape ``(n_wavenumber,)``.

        Raises
        ------
        ValueError
            If the data are 1-D (no scan dimension) or if no baseline is stored.
        IndexError
            If ``scan_index`` is outside ``[0, n_scan)``.
        """
        if self.ndim == 1:
            raise ValueError("get_baseline_scan requires 2-D data")
        if not self.has_baseline:
            raise ValueError("No baseline has been stored; run correct_baseline() first.")
        if not (0 <= scan_index < self.shape[0]):
            raise IndexError(
                f"scan_index {scan_index} out of bounds for {self.shape[0]} scans"
            )
        return self.ds["baseline"].isel(scan=scan_index).values

    def get_scan_by_tos(
        self,
        target_tos: Union[float, Sequence[float]],
        method: Literal["nearest", "linear"] = "nearest",
        tolerance_seconds: Optional[float] = 10,
    ) -> Union[npt.NDArray]:
        """Return the scan(s) nearest to one or more target elapsed-time values.

        Parameters
        ----------
        target_tos : float or sequence of float
            Target elapsed time(s) in seconds.
        method : {'nearest', 'linear'}, optional
            Interpolation method passed to ``xarray.DataArray.sel``
            (default is ``'nearest'``).
        tolerance_seconds : float or None, optional
            Maximum allowed distance between a target and the nearest scan.
            Raises ``ValueError`` when exceeded.  Pass ``None`` to disable
            (default is ``10``).

        Returns
        -------
        numpy.ndarray
            Shape ``(n_wavenumber,)`` for a scalar target or
            ``(n_targets, n_wavenumber)`` for a sequence.

        Raises
        ------
        ValueError
            If the data are 1-D, if no ``'tos'`` coordinate is present, or if
            any target exceeds ``tolerance_seconds`` from the nearest scan.
        """
        if self.ndim == 1:
            raise ValueError("get_scan_by_tos requires 2-D data")
        if self.tos is None:
            raise ValueError("get_scan_by_tos requires 'tos' coordinate")
        return self._select_var_by_tos(self.ds["data"], target_tos, method, tolerance_seconds)

    def get_baseline_by_tos(
        self,
        target_tos: Union[float, Sequence[float]],
        method: Literal["nearest", "linear"] = "nearest",
        tolerance_seconds: Optional[float] = 10,
    ) -> Union[npt.NDArray]:
        """Return the stored baseline curve(s) nearest to one or more target elapsed-time values.

        Parameters
        ----------
        target_tos : float or sequence of float
            Target elapsed time(s) in seconds.
        method : {'nearest', 'linear'}, optional
            Interpolation method passed to ``xarray.DataArray.sel``
            (default is ``'nearest'``).
        tolerance_seconds : float or None, optional
            Maximum allowed distance between a target and the nearest scan.
            Raises ``ValueError`` when exceeded.  Pass ``None`` to disable
            (default is ``10``).

        Returns
        -------
        numpy.ndarray
            Shape ``(n_wavenumber,)`` for a scalar target or
            ``(n_targets, n_wavenumber)`` for a sequence.

        Raises
        ------
        ValueError
            If the data are 1-D, if no baseline is stored, if no ``'tos'``
            coordinate is present, or if any target exceeds
            ``tolerance_seconds`` from the nearest scan.
        """
        if self.ndim == 1:
            raise ValueError("get_baseline_by_tos requires 2-D data")
        if not self.has_baseline:
            raise ValueError("No baseline has been stored; run correct_baseline() first.")
        if self.tos is None:
            raise ValueError("get_baseline_by_tos requires 'tos' coordinate")
        return self._select_var_by_tos(self.ds["baseline"], target_tos, method, tolerance_seconds)

    def get_id_by_tos(
        self,
        target_tos: Union[float, Sequence[float]],
        method: Literal["nearest", "linear"] = "nearest",
        tolerance_seconds: Optional[float] = 10,
    ) -> Union[int, npt.NDArray]:
        """Return the scan id(s) nearest to one or more target elapsed-time values.

        Parameters
        ----------
        target_tos : float or sequence of float
            Target elapsed time(s) in seconds.
        method : {'nearest', 'linear'}, optional
            Interpolation method passed to ``xarray.DataArray.sel``
            (default is ``'nearest'``).
        tolerance_seconds : float or None, optional
            Maximum allowed distance between a target and the nearest scan.
            Raises ``ValueError`` when exceeded.  Pass ``None`` to disable
            (default is ``10``).

        Returns
        -------
        int or numpy.ndarray
            Scan id (the ``'scan'`` coordinate value) for a scalar target,
            or an array of scan ids for a sequence of targets.

        Raises
        ------
        ValueError
            If the data are 1-D, if no ``'tos'`` coordinate is present, or if
            any target exceeds ``tolerance_seconds`` from the nearest scan.
        """
        if self.ndim == 1:
            raise ValueError("get_id_by_tos requires 2-D data")
        if self.tos is None:
            raise ValueError("get_id_by_tos requires 'tos' coordinate")

        scalar_input = np.ndim(target_tos) == 0
        targets = [float(target_tos)] if scalar_input else [float(t) for t in target_tos]

        def _id_for(t: float) -> int:
            if tolerance_seconds is not None:
                nearest_dist = float(np.abs(self.tos - t).min())
                if nearest_dist > tolerance_seconds:
                    raise ValueError(
                        f"Requested tos {t:.1f}s is {nearest_dist:.1f}s from the nearest scan "
                        f"(tolerance: {tolerance_seconds:.1f}s)"
                    )
            return int(self.ds["scan"].sel(tos=t, method=method).item())

        ids = np.array([_id_for(t) for t in targets], dtype=int)
        return int(ids[0]) if scalar_input else ids

    def get_scan_by_tos_average(
        self,
        target_tos: Union[float, Sequence[float]],
        method: Literal["nearest", "linear"] = "nearest",
        tolerance_seconds: Optional[float] = 10,
        number_of_scans: Optional[int] = None,
        time_window: Optional[float] = None,
        direction: Literal["forward", "backward", "center"] = "center",
    ) -> Union[npt.NDArray]:
        """Return the scan averaged over a window centred on each target tos.

        When neither ``number_of_scans`` nor ``time_window`` is supplied the
        single nearest scan is returned (equivalent to ``get_scan_by_tos``).

        Parameters
        ----------
        target_tos : float or sequence of float
            Target elapsed time(s) in seconds that anchor the averaging window.
        method : {'nearest', 'linear'}, optional
            Selection method for the anchor scan (default is ``'nearest'``).
        tolerance_seconds : float or None, optional
            Maximum distance in seconds between a target and its anchor scan
            (default is ``10``).
        number_of_scans : int or None, optional
            Number of consecutive scans to average.  Mutually exclusive with
            ``time_window``.
        time_window : float or None, optional
            Duration in seconds over which scans are averaged.  Mutually
            exclusive with ``number_of_scans``.
        direction : {'forward', 'backward', 'center'}, optional
            Position of the anchor within the window (default is
            ``'center'``).

        Returns
        -------
        numpy.ndarray
            Shape ``(n_wavenumber,)`` for a scalar target or
            ``(n_targets, n_wavenumber)`` for a sequence.

        Raises
        ------
        ValueError
            If the data are 1-D, if no ``'tos'`` coordinate is present, if
            any target exceeds the tolerance, or if the resulting window is
            empty.
        """
        if self.ndim == 1:
            raise ValueError("get_scan_by_tos_average requires 2-D data")
        if self.tos is None:
            raise ValueError("get_scan_by_tos_average requires 'tos' coordinate")
        if (number_of_scans is None) == (time_window is None):
            return self.get_scan_by_tos(target_tos, method=method, tolerance_seconds=tolerance_seconds)
            raise ValueError("Provide exactly one of: number_of_scans or time_window")

        scalar_input = np.ndim(target_tos) == 0
        targets = [float(target_tos)] if scalar_input else [float(t) for t in target_tos]
        tos_values = self.tos  # sorted 1-D array

        def _anchor_index(t: float) -> int:
            """Index of the scan nearest to t, with tolerance check."""
            dists = np.abs(tos_values - t)
            idx = int(dists.argmin())
            if tolerance_seconds is not None and dists[idx] > tolerance_seconds:
                raise ValueError(
                    f"Requested tos {t:.1f}s is {dists[idx]:.1f}s from the nearest scan "
                    f"(tolerance: {tolerance_seconds:.1f}s)"
                )
            return idx

        def _window_indices(anchor_idx: int) -> slice:
            """Return the index slice for the averaging window."""
            n = len(tos_values)

            if number_of_scans is not None:
                half = number_of_scans // 2
                if direction == "center":
                    i0 = anchor_idx - half
                    i1 = anchor_idx + (number_of_scans - half)  # handles odd counts correctly
                elif direction == "forward":
                    i0 = anchor_idx
                    i1 = anchor_idx + number_of_scans
                else:  # backward
                    i0 = anchor_idx - number_of_scans + 1
                    i1 = anchor_idx + 1

            else:  # time_window
                t_anchor = tos_values[anchor_idx]
                half_w = time_window / 2.0
                if direction == "center":
                    t0, t1 = t_anchor - half_w, t_anchor + half_w
                elif direction == "forward":
                    t0, t1 = t_anchor, t_anchor + time_window
                else:  # backward
                    t0, t1 = t_anchor - time_window, t_anchor
                i0 = int(np.searchsorted(tos_values, t0, side="left"))
                i1 = int(np.searchsorted(tos_values, t1, side="right"))

            i0 = max(i0, 0)
            i1 = min(i1, n)

            if i0 >= i1:
                raise ValueError(
                    f"Window [{i0}:{i1}] is empty for anchor index {anchor_idx}. "
                    "Check number_of_scans / time_window against the data range."
                )
            return slice(i0, i1)

        def _average_one(t: float) -> npt.NDArray:
            anchor_idx = _anchor_index(t)
            win = _window_indices(anchor_idx)
            window_data = self.ds["data"].isel(scan=win).values  # shape: (n_scans_in_window, n_masses)
            return window_data.mean(axis=0)

        results = np.vstack([_average_one(t) for t in targets])
        return results[0] if scalar_input else results

    def get_evolution(
        self,
        wavenumber_per_cm: Union[float, list[float], npt.NDArray],
        method: Literal["nearest", "linear"] = "nearest",
        tolerance_per_cm: Optional[float] = None,
        rolling_window: Optional[int] = None,
    ) -> xr.DataArray:
        """Return the intensity evolution over scans at one or more wavenumbers.

        Parameters
        ----------
        wavenumber_per_cm : float or array-like
            Target wavenumber(s) in cm⁻¹.
        method : {'nearest', 'linear'}, optional
            Selection method passed to ``xarray.DataArray.sel``
            (default is ``'nearest'``).
        tolerance_per_cm : float or None, optional
            Maximum allowed distance in cm⁻¹ between a target and the nearest
            grid point.  Raises ``ValueError`` when exceeded.  Pass ``None``
            to disable (default is ``None``).
        rolling_window : int or None, optional
            If provided, apply a rolling mean over ``rolling_window`` scans
            (centred, minimum one period) before returning (default is
            ``None``).

        Returns
        -------
        xr.DataArray
            DataArray with dims ``('scan',)`` or ``('scan', 'wavenumber')``
            depending on the number of target wavenumbers.

        Raises
        ------
        ValueError
            If the data are 1-D or if a target wavenumber exceeds
            ``tolerance_per_cm`` from the nearest grid point.
        """
        if self.ndim == 1:
            raise ValueError("get_evolution requires 2-D data")

        targets_si = np.atleast_1d(np.asarray(wavenumber_per_cm, dtype=float)) * 100.0

        if tolerance_per_cm is not None:
            tol_si = tolerance_per_cm * 100.0
            for t in targets_si:
                nearest_dist = float(np.abs(self.wavenumber - t).min())
                if nearest_dist > tol_si:
                    raise ValueError(
                        f"Requested wavenumber {t / 100:.1f} cm⁻¹ is "
                        f"{nearest_dist / 100:.1f} cm⁻¹ from the nearest grid point "
                        f"(tolerance: {tolerance_per_cm:.1f} cm⁻¹)"
                    )

        result = self.ds["data"].sel(wavenumber=targets_si, method=method)

        if rolling_window is not None:
            result = result.rolling(scan=rolling_window, center=True, min_periods=1).mean()

        return result

    # ----------------------------------------------------------------
    # Immutable — background
    # ----------------------------------------------------------------

    def with_background(
        self,
        background: Union[npt.NDArray, "IRData"],
        data_type: Optional[IRDataType] = None,
    ) -> "IRData":
        """Set or switch the background spectrum (always stored as single_beam).

        If no background is currently assigned: assigns it without touching data values.
        If a background is already assigned: converts the new background to single_beam
        using the existing background, then recalculates data values accordingly.

        Parameters
        ----------
        background : numpy.ndarray or IRData
            Background spectrum.  If an ``IRData`` its ``data_type`` is used
            unless overridden by ``data_type``.
        data_type : str or None, optional
            Override the data type of ``background`` when it is a plain array.
            Must be one of the ``IRDataType`` literals (default is ``None``).

        Returns
        -------
        IRData
            New instance with the background set and data values recalculated
            if a previous background existed.

        Raises
        ------
        ValueError
            If a 2-D ``IRData`` is passed as background, if the background
            array is not 1-D, or if the background size does not match the
            wavenumber axis.
        """
        if isinstance(background, IRData):
            if background.ndim != 1:
                raise ValueError("Background IRData must be 1-D (a single spectrum)")
            bg_values = background.values
            bg_data_type = data_type if data_type is not None else background.data_type
        else:
            bg_values = np.asarray(background, dtype=float)
            if bg_values.ndim != 1:
                raise ValueError("Background array must be 1-D")
            bg_data_type = data_type

        if bg_values.size != self.wavenumber.size:
            raise ValueError(
                f"Background size ({bg_values.size}) does not match wavenumber axis ({self.wavenumber.size})"
            )

        bg_sb = self._bg_to_single_beam(bg_values, bg_data_type)

        if not self.has_background:
            return self._set_background(bg_sb)
        else:
            return self._switch_background(bg_sb)

    def with_background_scan(self, scan_index: int) -> "IRData":
        """Use the scan at scan_index as the new background.

        The scan is extracted as a 1-D IRData with the same data_type as self.
        If data is not single_beam, the scan is converted back to single_beam using
        the existing background before being assigned.

        Parameters
        ----------
        scan_index : int
            Zero-based index of the scan to promote to background.

        Returns
        -------
        IRData
            New instance with the selected scan set as background.

        Raises
        ------
        ValueError
            If the data are 1-D.
        """
        if self.ndim == 1:
            raise ValueError("with_background_scan requires 2-D data")
        return self.with_background(self.select_by_idx(scan_index))

    def with_background_by_tos(
        self,
        target_tos: float,
        method: Literal["nearest", "linear"] = "nearest",
        tolerance_seconds: Optional[float] = 10,
    ) -> "IRData":
        """Use the scan nearest to target_tos as the new background.

        Same single_beam conversion logic as with_background_scan.

        Parameters
        ----------
        target_tos : float
            Target elapsed time in seconds.
        method : {'nearest', 'linear'}, optional
            Selection method (default is ``'nearest'``).
        tolerance_seconds : float or None, optional
            Maximum distance in seconds from the target to the nearest scan
            (default is ``10``).

        Returns
        -------
        IRData
            New instance with the selected scan set as background.

        Raises
        ------
        ValueError
            If the data are 1-D or if the target exceeds the tolerance.
        """
        if self.ndim == 1:
            raise ValueError("with_background_by_tos requires 2-D data")
        return self.with_background(self.select_by_tos(target_tos, method=method, tolerance_seconds=tolerance_seconds))

    def del_background(self) -> "IRData":
        """Remove the background spectrum without changing data values.

        Returns
        -------
        IRData
            New instance with the ``'background'`` variable dropped.
        """
        return self._del_background()

    def set_background(
        self,
        background: Union[npt.NDArray, "IRData"],
        data_type: Optional[IRDataType] = None,
    ) -> "IRData":
        """Force-assign a background without recalculating values (drops any existing background first).

        Use with_background() instead when switching backgrounds should trigger recalculation.

        Parameters
        ----------
        background : numpy.ndarray or IRData
            Background spectrum to store.  If an ``IRData`` its ``data_type``
            is used unless overridden by ``data_type``.
        data_type : str or None, optional
            Override the data type of ``background`` when it is a plain array
            (default is ``None``).

        Returns
        -------
        IRData
            New instance with the background assigned and data values
            unchanged.

        Raises
        ------
        ValueError
            If a 2-D ``IRData`` is passed, if the array is not 1-D, or if the
            background size does not match the wavenumber axis.
        """
        if isinstance(background, IRData):
            if background.ndim != 1:
                raise ValueError("Background IRData must be 1-D (a single spectrum)")
            bg_values = background.values
            bg_data_type = data_type if data_type is not None else background.data_type
        else:
            bg_values = np.asarray(background, dtype=float)
            if bg_values.ndim != 1:
                raise ValueError("Background array must be 1-D")
            bg_data_type = data_type

        if bg_values.size != self.wavenumber.size:
            raise ValueError(
                f"Background size ({bg_values.size}) does not match wavenumber axis ({self.wavenumber.size})"
            )

        bg_sb = self._bg_to_single_beam(bg_values, bg_data_type)
        return self._del_background()._set_background(bg_sb)

    # ----------------------------------------------------------------
    # Immutable — baseline
    # ----------------------------------------------------------------

    def del_baseline(self) -> "IRData":
        """Remove the stored baseline curve without changing data values.

        Returns
        -------
        IRData
            New instance with the ``'baseline'`` variable dropped.
        """
        return self._del_baseline()

    def unbaseline(self) -> "IRData":
        """Return a new instance with the baseline added back into ``data`` and dropped.

        Lets the regular getters (``get_scan``, ``get_scan_by_tos``,
        ``get_evolution``, ...) return pre-correction values, e.g.
        ``ir.unbaseline().get_scan_by_tos(120)``.

        Returns
        -------
        IRData
            New instance with ``data`` equal to ``data_unbaselined`` and no
            stored baseline.  Returns ``self`` unchanged if no baseline is
            stored.
        """
        if not self.has_baseline:
            return self
        ds_new = self._build_ds(
            self.wavenumber, self.data_unbaselined, tos=self.tos, attrs=dict(self.ds.attrs)
        )
        return IRData(ds=self._carry_background(ds_new))

    # ----------------------------------------------------------------
    # Immutable — selection and sorting
    # ----------------------------------------------------------------

    def assign_tos_start(self, tos_start: Union[pd.Timestamp, str]) -> "IRData":
        """Return a new instance with ``tos_start`` replaced and ``tos`` shifted accordingly.

        Parameters
        ----------
        tos_start : pandas.Timestamp or str
            New absolute start time.  Strings are parsed by
            ``pandas.Timestamp``.

        Returns
        -------
        IRData
            New instance with updated ``tos_start`` attribute and adjusted
            ``tos`` coordinate.
        """
        old_tos_start = self.tos_start
        new_tos_start = pd.Timestamp(tos_start)

        attrs = dict(self.ds.attrs)
        attrs["tos_start"] = new_tos_start.isoformat()

        old_tos = self.tos
        new_tos = old_tos + (new_tos_start - old_tos_start).total_seconds() if old_tos is not None else None

        ds = self._build_ds(
            wavenumber_si=self.wavenumber,
            values=self.values,
            tos=new_tos,
            attrs=attrs,
        )
        return IRData(ds=self._carry_baseline(self._carry_background(ds)))

    def sort(self, by: str | Sequence[str] = "wavenumber", ascending: bool = True) -> "IRData":
        """Return a new instance with coordinates sorted.

        Parameters
        ----------
        by : str or sequence of str, optional
            Coordinate name(s) to sort by (default is ``'wavenumber'``).
        ascending : bool, optional
            Sort in ascending order when ``True`` (default is ``True``).

        Returns
        -------
        IRData
            New instance with the dataset sorted along the specified
            coordinate(s).
        """
        ds_sorted = self.ds.sortby(by, ascending=ascending)
        return IRData(ds=ds_sorted)

    def select_by_idx(self, idx: int) -> "IRData":
        """Return a 1-D IRData containing the scan at the given integer index.

        Parameters
        ----------
        idx : int
            Zero-based scan index.

        Returns
        -------
        IRData
            New 1-D instance for the selected scan.

        Raises
        ------
        ValueError
            If the data are already 1-D.
        IndexError
            If ``idx`` is outside ``[0, n_scan)``.
        """
        if self.ndim == 1:
            raise ValueError("select_by_idx requires 2-D data")
        if not (0 <= idx < self.shape[0]):
            raise IndexError(f"idx {idx} out of bounds for {self.shape[0]} scans")
        ds_selected = self.ds.isel(scan=idx)
        return IRData(ds=ds_selected)

    def select_by_tos(self, target_tos: float, method: Literal["nearest", "linear"] = "nearest", tolerance_seconds: Optional[float] = 10) -> "IRData":
        """Return a 1-D IRData containing the scan nearest to ``target_tos``.

        Parameters
        ----------
        target_tos : float
            Target elapsed time in seconds.
        method : {'nearest', 'linear'}, optional
            Selection method (default is ``'nearest'``).
        tolerance_seconds : float or None, optional
            Maximum distance in seconds from the target to the nearest scan.
            Raises ``ValueError`` when exceeded (default is ``10``).

        Returns
        -------
        IRData
            New 1-D instance for the selected scan.

        Raises
        ------
        ValueError
            If the data are 1-D, if no ``'tos'`` coordinate is present, or if
            the target exceeds the tolerance.
        """
        if self.ndim == 1:
            raise ValueError("select_by_tos requires 2-D data")
        if self.tos is None:
            raise ValueError("select_by_tos requires 'tos' coordinate")

        if tolerance_seconds is not None:
            nearest_dist = float(np.abs(self.tos - target_tos).min())
            if nearest_dist > tolerance_seconds:
                raise ValueError(
                    f"Requested tos {target_tos:.1f}s is {nearest_dist:.1f}s from the nearest scan "
                    f"(tolerance: {tolerance_seconds:.1f}s)"
                )

        ds_selected = self.ds.sel(tos=target_tos, method=method)
        return IRData(ds=ds_selected)

    def select_wavenumber_range(
        self,
        min_cm: Optional[float] = None,
        max_cm: Optional[float] = None,
    ) -> "IRData":
        """Return a new instance restricted to a wavenumber sub-range.

        Parameters
        ----------
        min_cm : float or None, optional
            Lower bound in cm⁻¹ (inclusive).  ``None`` means no lower bound.
        max_cm : float or None, optional
            Upper bound in cm⁻¹ (inclusive).  ``None`` means no upper bound.

        Returns
        -------
        IRData
            New instance with the wavenumber axis truncated and the background
            sliced to match.
        """
        da = self.ds["data"]
        if min_cm is not None:
            wn = da.coords["wavenumber"].values
            da = da.sel(wavenumber=wn >= min_cm * 100.0)
        if max_cm is not None:
            wn = da.coords["wavenumber"].values
            da = da.sel(wavenumber=wn <= max_cm * 100.0)

        ds_new = self._build_ds(
            wavenumber_si=da.coords["wavenumber"].values,
            values=da.values,
            tos=da.coords["tos"].values if "tos" in da.coords else None,
            attrs=dict(self.ds.attrs),
        )
        ds_new = self._slice_background_to(ds_new)
        ds_new = self._slice_baseline_to(ds_new)
        return IRData(ds=ds_new)

    def select_wavenumber_index_range(
        self,
        min_idx: Optional[int] = None,
        max_idx: Optional[int] = None,
    ) -> "IRData":
        """Return a new instance restricted to a wavenumber sub-range selected by position.

        Unlike ``select_wavenumber_range``, bounds are given as zero-based
        positional indices into the wavenumber axis rather than cm⁻¹ values.

        Parameters
        ----------
        min_idx : int or None, optional
            Lower bound index (inclusive).  ``None`` means index ``0``.
        max_idx : int or None, optional
            Upper bound index (inclusive).  ``None`` means the last index.

        Returns
        -------
        IRData
            New instance with the wavenumber axis truncated and the background
            sliced to match.

        Raises
        ------
        IndexError
            If ``min_idx`` or ``max_idx`` is outside ``[0, n_wavenumber)``.
        """
        n = self.shape[-1]
        lo = 0 if min_idx is None else min_idx
        hi = n - 1 if max_idx is None else max_idx
        if not (0 <= lo < n) or not (0 <= hi < n):
            raise IndexError(f"index range [{lo}, {hi}] out of bounds for {n} wavenumber points")

        da = self.ds["data"].isel(wavenumber=slice(lo, hi + 1))
        ds_new = self._build_ds(
            wavenumber_si=da.coords["wavenumber"].values,
            values=da.values,
            tos=da.coords["tos"].values if "tos" in da.coords else None,
            attrs=dict(self.ds.attrs),
        )
        ds_new = self._slice_background_to(ds_new)
        ds_new = self._slice_baseline_to(ds_new)
        return IRData(ds=ds_new)

    def select_tos_range(
        self,
        min_s: Optional[float] = None,
        max_s: Optional[float] = None,
    ) -> "IRData":
        """Return a new instance restricted to scans within a tos sub-range.

        Parameters
        ----------
        min_s : float or None, optional
            Lower bound in seconds (inclusive).  ``None`` means no lower bound.
        max_s : float or None, optional
            Upper bound in seconds (inclusive).  ``None`` means no upper bound.

        Returns
        -------
        IRData
            New instance containing only the scans within the specified range.

        Raises
        ------
        ValueError
            If no ``'tos'`` coordinate is present.
        """
        if self.tos is None:
            raise ValueError("select_tos_range requires a 'tos' coordinate")

        # Operate on the whole Dataset so 'background' and 'baseline' (if present) are
        # carried through the scan-axis selection automatically.
        ds_new = self.ds
        if min_s is not None:
            tos = ds_new.coords["tos"].values
            if not np.any(tos >= min_s):
                min_s = tos[0]
                logger.warning(f"min_s {min_s:.1f}s is greater than all 'tos' values; using min_s={min_s:.1f}s instead")
            ds_new = ds_new.isel(scan=tos >= min_s)
        if max_s is not None:
            tos = ds_new.coords["tos"].values
            if not np.any(tos <= max_s):
                max_s = tos[-1]
                logger.warning(f"max_s {max_s:.1f}s is less than all 'tos' values; using max_s={max_s:.1f}s instead")
            ds_new = ds_new.isel(scan=tos <= max_s)

        # tos values are absolute elapsed seconds, so tos_start + tos[i] remains valid
        return IRData(ds=ds_new)

    def select_scan_id_range(
        self,
        min_id: Optional[int] = None,
        max_id: Optional[int] = None,
    ) -> "IRData":
        """Return a new instance restricted to scans within a scan-id sub-range.

        Unlike ``select_tos_range``, bounds are given as scan ids (the
        ``'scan'`` coordinate, see ``get_id_by_tos``) rather than elapsed time.

        Parameters
        ----------
        min_id : int or None, optional
            Lower bound on the scan id (inclusive).  ``None`` means no lower bound.
        max_id : int or None, optional
            Upper bound on the scan id (inclusive).  ``None`` means no upper bound.

        Returns
        -------
        IRData
            New instance containing only the scans within the specified id range.

        Raises
        ------
        ValueError
            If the data are 1-D (no scan dimension).
        """
        if self.ndim == 1:
            raise ValueError("select_scan_id_range requires 2-D data")

        ds_new = self.ds
        if min_id is not None:
            scan_ids = ds_new.coords["scan"].values
            if not np.any(scan_ids >= min_id):
                min_id = int(scan_ids[0])
                logger.warning(f"min_id {min_id} is greater than all scan ids; using min_id={min_id} instead")
            ds_new = ds_new.isel(scan=scan_ids >= min_id)
        if max_id is not None:
            scan_ids = ds_new.coords["scan"].values
            if not np.any(scan_ids <= max_id):
                max_id = int(scan_ids[-1])
                logger.warning(f"max_id {max_id} is less than all scan ids; using max_id={max_id} instead")
            ds_new = ds_new.isel(scan=scan_ids <= max_id)

        return IRData(ds=ds_new)

    # ----------------------------------------------------------------
    # Immutable — smoothing
    # ----------------------------------------------------------------

    def smooth_savgol(self, window_length: int = 21, polyorder: int = 3) -> "IRData":
        """Apply a Savitzky-Golay smoothing filter along the wavenumber axis.

        Parameters
        ----------
        window_length : int, optional
            Length of the filter window in points (default is ``21``).
        polyorder : int, optional
            Polynomial order for the filter (default is ``3``).

        Returns
        -------
        IRData
            New instance with smoothed spectral values.
        """
        from scipy.signal import savgol_filter

        def _filter(arr: npt.NDArray) -> npt.NDArray:
            if arr.ndim == 1:
                return savgol_filter(arr, window_length, polyorder)
            return np.apply_along_axis(lambda m: savgol_filter(m, window_length, polyorder), axis=1, arr=arr)

        smoothed = _filter(self.values)
        smoothed_baseline = _filter(self.baseline) if self.has_baseline else None
        ds_new = self._build_ds(self.wavenumber, smoothed, tos=self.tos, attrs=dict(self.ds.attrs))
        ds_new = self._with_baseline(ds_new, smoothed_baseline)
        return IRData(ds=self._carry_background(ds_new))

    def smooth_gaussian(self, sigma_cm: float) -> "IRData":
        """Apply a Gaussian smoothing filter along the wavenumber axis.

        Parameters
        ----------
        sigma_cm : float
            Standard deviation of the Gaussian kernel in cm⁻¹.

        Returns
        -------
        IRData
            New instance with smoothed spectral values.
        """
        from scipy.ndimage import gaussian_filter1d

        sigma_si = sigma_cm * 100.0

        def _filter(arr: npt.NDArray) -> npt.NDArray:
            if arr.ndim == 1:
                return gaussian_filter1d(arr, sigma=sigma_si)
            return np.apply_along_axis(lambda m: gaussian_filter1d(m, sigma=sigma_si), axis=1, arr=arr)

        smoothed = _filter(self.values)
        smoothed_baseline = _filter(self.baseline) if self.has_baseline else None
        ds_new = self._build_ds(self.wavenumber, smoothed, tos=self.tos, attrs=dict(self.ds.attrs))
        ds_new = self._with_baseline(ds_new, smoothed_baseline)
        return IRData(ds=self._carry_background(ds_new))

    def smooth_moving(self, window_size: int = 5) -> "IRData":
        """Apply a uniform moving-average filter along the wavenumber axis.

        Parameters
        ----------
        window_size : int, optional
            Number of points in the averaging window (default is ``5``).

        Returns
        -------
        IRData
            New instance with smoothed spectral values.

        Raises
        ------
        ValueError
            If ``window_size`` is less than ``1``.
        """
        if window_size < 1:
            raise ValueError("window_size must be >= 1")

        kernel = np.ones(window_size) / window_size

        def _filter(arr: npt.NDArray) -> npt.NDArray:
            if arr.ndim == 1:
                return np.convolve(arr, kernel, mode="same")
            return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=arr)

        smoothed = _filter(self.values)
        smoothed_baseline = _filter(self.baseline) if self.has_baseline else None
        ds_new = self._build_ds(self.wavenumber, smoothed, tos=self.tos, attrs=dict(self.ds.attrs))
        ds_new = self._with_baseline(ds_new, smoothed_baseline)
        return IRData(ds=self._carry_background(ds_new))

    # ----------------------------------------------------------------
    # Immutable — baseline correction
    # ----------------------------------------------------------------

    def correct_offset(
        self,
        anchor_range_cm: Tuple[float, float] = (2500, 2600),
    ) -> "IRData":
        """Subtract a constant offset computed as the mean over an anchor range.

        Parameters
        ----------
        anchor_range_cm : tuple of float, optional
            ``(low, high)`` wavenumber bounds in cm⁻¹ that define the
            baseline anchor region (default is ``(2500, 2600)``).

        Returns
        -------
        IRData
            New instance with the offset removed.  The subtracted offset is
            added to (or stored as) ``ds['baseline']``; recover the
            pre-correction values via ``data_unbaselined``.

        Raises
        ------
        ValueError
            If no wavenumber points fall within the anchor range.
        """
        lo_si = min(anchor_range_cm) * 100.0
        hi_si = max(anchor_range_cm) * 100.0
        wn = self.wavenumber
        mask = (wn >= lo_si) & (wn <= hi_si)
        if not mask.any():
            raise ValueError(
                f"No wavenumber points in anchor range "
                f"{min(anchor_range_cm):.0f}–{max(anchor_range_cm):.0f} cm⁻¹"
            )

        if self.ndim == 1:
            offset = self.values[mask].mean()
            offset_arr = np.full_like(self.values, offset)
        else:
            offset = self.values[:, mask].mean(axis=1, keepdims=True)
            offset_arr = np.broadcast_to(offset, self.values.shape).copy()
        corrected = self.values - offset_arr

        new_baseline = self._accumulate_baseline(offset_arr)
        new_attrs = {**self.ds.attrs, "baseline_anchor_range_cm": list(anchor_range_cm)}
        ds_new = self._build_ds(wn, corrected, tos=self.tos, attrs=new_attrs)
        ds_new = self._with_baseline(ds_new, new_baseline)
        return IRData(ds=self._carry_background(ds_new))

    def correct_pchip(
        self,
        control_points_cm: Sequence[float],
        point_avg_half_width: int = 0,
    ) -> "IRData":
        """Subtract a PCHIP-interpolated baseline through the given control points.

        Parameters
        ----------
        control_points_cm : sequence of float
            Wavenumber positions in cm⁻¹ used as knots for the PCHIP spline.
        point_avg_half_width : int, optional
            Half-width (in points) of the local averaging window used to
            compute the intensity at each knot (default is ``0``, i.e. the
            single nearest point).

        Returns
        -------
        IRData
            New instance with the PCHIP baseline subtracted.  The subtracted
            curve is added to (or stored as) ``ds['baseline']``; recover the
            pre-correction values via ``data_unbaselined``.
        """
        from scipy.interpolate import PchipInterpolator

        wn_si = self.wavenumber
        wn_cm = wn_si / 100.0
        cps = np.sort(np.asarray(control_points_cm, dtype=float))

        def _pchip_curve(spectrum_1d: np.ndarray) -> np.ndarray:
            x_knots = np.empty(len(cps))
            y_knots = np.empty(len(cps))
            for j, cp_cm in enumerate(cps):
                idx = int(np.abs(wn_cm - cp_cm).argmin())
                lo = max(0, idx - point_avg_half_width)
                hi = min(len(spectrum_1d), idx + point_avg_half_width + 1)
                x_knots[j] = wn_cm[idx]
                y_knots[j] = spectrum_1d[lo:hi].mean()
            return PchipInterpolator(x_knots, y_knots)(wn_cm)

        if self.ndim == 1:
            pchip_curve = _pchip_curve(self.values)
        else:
            pchip_curve = np.apply_along_axis(_pchip_curve, axis=1, arr=self.values)
        corrected = self.values - pchip_curve

        new_baseline = self._accumulate_baseline(pchip_curve)
        new_attrs = {
            **self.ds.attrs,
            "baseline_pchip_control_points_cm": sorted(control_points_cm),
            "baseline_pchip_half_width": point_avg_half_width,
        }
        ds_new = self._build_ds(wn_si, corrected, tos=self.tos, attrs=new_attrs)
        ds_new = self._with_baseline(ds_new, new_baseline)
        return IRData(ds=self._carry_background(ds_new))

    def correct_baseline(
        self,
        anchor_range_cm: Tuple[float, float] = (2500, 2600),
        control_points_cm: Optional[Sequence[float]] = None,
        point_avg_half_width: int = 0,
        double_offset: bool = True,
    ) -> "IRData":
        """Apply a two-step baseline correction: offset then optional PCHIP.

        Step 1 subtracts a constant offset via ``correct_offset``.  If
        ``control_points_cm`` is provided, step 2 removes a PCHIP spline via
        ``correct_pchip``, followed optionally by a second offset step (mirrors
        DRIFTS behaviour).

        Parameters
        ----------
        anchor_range_cm : tuple of float, optional
            Anchor wavenumber range in cm⁻¹ used for the offset step(s)
            (default is ``(2500, 2600)``).
        control_points_cm : sequence of float or None, optional
            Knot positions in cm⁻¹ for the PCHIP step.  ``None`` skips the
            PCHIP step (default is ``None``).
        point_avg_half_width : int, optional
            Half-width in points for local averaging at each PCHIP knot
            (default is ``0``).
        double_offset : bool, optional
            Apply a second offset correction after the PCHIP step when
            ``True`` (default is ``True``).

        Returns
        -------
        IRData
            New instance with the baseline removed.  The cumulative baseline
            curve is stored in ``ds['baseline']``; recover the
            pre-correction values via ``data_unbaselined``.
        """
        # Step 1: offset, step 2: PCHIP, step 3: optional second offset (mirrors DRIFTS behaviour)
        result = self.correct_offset(anchor_range_cm)
        if control_points_cm:
            result = result.correct_pchip(control_points_cm, point_avg_half_width)
            if double_offset:
                result = result.correct_offset(anchor_range_cm)
        return result

    def reapply_baseline(self) -> "IRData":
        """Re-run baseline correction using parameters stored in dataset attributes.

        Returns
        -------
        IRData
            New instance with the stored baseline correction reapplied.

        Raises
        ------
        ValueError
            If ``ds.attrs`` does not contain ``'baseline_anchor_range_cm'``.
        """
        # Re-runs correction using parameters stored in attributes (e.g. after average_scans)
        anchor_range_cm = self.ds.attrs.get("baseline_anchor_range_cm")
        if anchor_range_cm is None:
            raise ValueError("No baseline parameters found in attributes.")

        # Restore the unbaselined values first so correct_baseline() isn't applied on top
        # of the previously stored baseline (which would double-correct).
        base = self
        if self.has_baseline:
            restored_ds = self._build_ds(
                self.wavenumber, self.data_unbaselined, tos=self.tos, attrs=dict(self.ds.attrs)
            )
            base = IRData(ds=self._carry_background(restored_ds))

        return base.correct_baseline(
            anchor_range_cm=tuple(anchor_range_cm),
            control_points_cm=self.ds.attrs.get("baseline_pchip_control_points_cm"),
            point_avg_half_width=self.ds.attrs.get("baseline_pchip_half_width", 0),
        )


    # ----------------------------------------------------------------
    # Immutable — averaging
    # ----------------------------------------------------------------

    def average_scans(
        self,
        number_of_scans: int,
        tos_method: Literal["mean", "median", "first", "last"] = "first",
    ) -> "IRData":
        """Co-add consecutive groups of scans by averaging.

        Trailing scans that do not fill a complete group are discarded.

        Parameters
        ----------
        number_of_scans : int
            Number of consecutive scans per group.
        tos_method : {'mean', 'median', 'first', 'last'}, optional
            How to assign a representative ``tos`` value to each averaged
            group (default is ``'first'``).

        Returns
        -------
        IRData
            New instance with ``n_scan // number_of_scans`` averaged scans.

        Raises
        ------
        ValueError
            If the data are 1-D or if ``number_of_scans`` is less than ``1``.
        """
        if self.ndim == 1:
            raise ValueError("average_scans requires 2-D data")
        if number_of_scans < 1:
            raise ValueError("number_of_scans must be >= 1")

        n_averaged = self.shape[0] // number_of_scans
        new_values = (
            self.values[: n_averaged * number_of_scans]
            .reshape(n_averaged, number_of_scans, -1)
            .mean(axis=1)
        )
        new_baseline = None
        if self.has_baseline:
            new_baseline = (
                self.baseline[: n_averaged * number_of_scans]
                .reshape(n_averaged, number_of_scans, -1)
                .mean(axis=1)
            )

        new_tos = None
        if self.tos is not None:
            tos_blocks = self.tos[: n_averaged * number_of_scans].reshape(n_averaged, number_of_scans)
            if tos_method == "mean":
                new_tos = tos_blocks.mean(axis=1)
            elif tos_method == "median":
                new_tos = np.median(tos_blocks, axis=1)
            elif tos_method == "first":
                new_tos = tos_blocks[:, 0]
            elif tos_method == "last":
                new_tos = tos_blocks[:, -1]

        # tos values remain absolute elapsed seconds, so tos_start stays valid
        ds_new = self._build_ds(self.wavenumber, new_values, tos=new_tos, attrs=dict(self.ds.attrs))
        ds_new = self._with_baseline(ds_new, new_baseline)
        return IRData(ds=self._carry_background(ds_new))

    def average_scans_by_tos(
        self,
        target_tos: Union[float, Sequence[float]],
        method: Literal["nearest", "linear"] = "nearest",
        tolerance_seconds: Optional[float] = 10,
        number_of_scans: Optional[int] = None,
        time_window: Optional[float] = None,
        direction: Literal["forward", "backward", "center"] = "center",
    ) -> "IRData":
        """Return a new IRData from scans averaged around each target tos.

        Each target produces one averaged spectrum.  The anchor ``tos`` of the
        nearest real scan is used as the ``tos`` coordinate for that output
        spectrum.

        Parameters
        ----------
        target_tos : float or sequence of float
            Target elapsed time(s) in seconds.
        method : {'nearest', 'linear'}, optional
            Selection method for the anchor scan (default is ``'nearest'``).
        tolerance_seconds : float or None, optional
            Maximum distance in seconds between a target and its anchor scan
            (default is ``10``).
        number_of_scans : int or None, optional
            Number of scans to average.  Mutually exclusive with
            ``time_window``.
        time_window : float or None, optional
            Duration in seconds over which scans are averaged.  Mutually
            exclusive with ``number_of_scans``.
        direction : {'forward', 'backward', 'center'}, optional
            Position of the anchor within the window (default is
            ``'center'``).

        Returns
        -------
        IRData
            New instance with one scan per target tos.

        Raises
        ------
        ValueError
            If the data are 1-D or if any target exceeds the tolerance.
        """
        if self.ndim == 1:
            raise ValueError("average_scans_by_tos requires 2-D data")

        scalar_input = np.ndim(target_tos) == 0
        targets = [float(target_tos)] if scalar_input else [float(t) for t in target_tos]

        # Reuse the averaging logic — returns list of 1-D arrays (or one array if scalar)
        averaged = self.get_scan_by_tos_average(
            target_tos=target_tos,
            method=method,
            tolerance_seconds=tolerance_seconds,
            number_of_scans=number_of_scans,
            time_window=time_window,
            direction=direction,
        )

        # Normalise to list of 1-D arrays regardless of scalar/array input
        if scalar_input:
            averaged_list = [averaged]
        else:
            averaged_list = averaged  # already a list

        new_values = np.vstack(averaged_list)  # (n_targets, n_wavenumber)

        new_baseline = None
        if self.has_baseline:
            baseline_ir = IRData(ds=self._build_ds(self.wavenumber, self.baseline, tos=self.tos, attrs={}))
            bl_averaged = baseline_ir.get_scan_by_tos_average(
                target_tos=target_tos,
                method=method,
                tolerance_seconds=tolerance_seconds,
                number_of_scans=number_of_scans,
                time_window=time_window,
                direction=direction,
            )
            bl_list = [bl_averaged] if scalar_input else bl_averaged
            new_baseline = np.vstack(bl_list)

        # Anchor tos: the nearest actual tos to each target becomes the new coord
        tos_values = self.tos
        new_tos = np.array(targets)

        new_attrs = {
            **self.ds.attrs,
            "averaged_target_tos": [float(t) for t in targets],
            "averaged_anchor_tos": [
                float(tos_values[int(np.abs(tos_values - t).argmin())])
                for t in targets
            ],
            "averaged_direction": direction,
            "averaged_number_of_scans": number_of_scans,
            "averaged_time_window": time_window,
        }

        ds_new = self._build_ds(
            wavenumber_si=self.wavenumber,
            values=new_values,
            tos=new_tos,
            attrs=new_attrs,
        )
        ds_new = self._with_baseline(ds_new, new_baseline)
        return IRData(ds=self._carry_background(ds_new))

    # ----------------------------------------------------------------
    # Immutable - Normalisation
    # ----------------------------------------------------------------

    def normalise_max(self) -> "IRData":
        """Divide all values by the global maximum.

        Returns
        -------
        IRData
            New instance with values in ``[0, 1]``.  Returns ``self``
            unchanged if the maximum is zero.
        """
        max_val = self.values.max()
        if max_val == 0:
            logger.warning("Maximum value is zero; returning original data without normalisation")
            return self
        new_values = self.values / max_val
        new_baseline = self.baseline / max_val if self.has_baseline else None
        ds_new = self._build_ds(self.wavenumber, new_values, tos=self.tos, attrs=dict(self.ds.attrs))
        ds_new = self._with_baseline(ds_new, new_baseline)
        return IRData(ds=self._carry_background(ds_new))

    def normalise_integral(self) -> "IRData":
        """Divide each spectrum by its trapezoidal integral over the wavenumber axis.

        Returns
        -------
        IRData
            New instance with unit-integral spectra.  Returns ``self``
            unchanged if any integral is zero.
        """
        integral = np.trapz(self.values, x=self.wavenumber, axis=-1)
        if np.any(integral == 0):
            logger.warning("Integral is zero for some scans; returning original data without normalisation")
            return self
        new_values = self.values / integral[..., np.newaxis]
        new_baseline = self.baseline / integral[..., np.newaxis] if self.has_baseline else None
        ds_new = self._build_ds(self.wavenumber, new_values, tos=self.tos, attrs=dict(self.ds.attrs))
        ds_new = self._with_baseline(ds_new, new_baseline)
        return IRData(ds=self._carry_background(ds_new))

    def normalise_reference(self, reference: npt.NDArray) -> "IRData":
        """Divide all spectra element-wise by a reference spectrum.

        Parameters
        ----------
        reference : numpy.ndarray
            1-D reference spectrum with the same number of wavenumber points.

        Returns
        -------
        IRData
            New instance with values divided by ``reference``.  Returns
            ``self`` unchanged if ``reference`` contains any zeros.

        Raises
        ------
        ValueError
            If ``reference`` is not 1-D or its size does not match the
            wavenumber axis.
        """
        if reference.ndim != 1:
            raise ValueError("Reference spectrum must be 1-D")
        if reference.size != self.wavenumber.size:
            raise ValueError(f"Reference size ({reference.size}) does not match wavenumber size ({self.wavenumber.size})")
        if np.any(reference == 0):
            logger.warning("Reference spectrum contains zero values; returning original data without normalisation")
            return self

        new_values = self.values / reference
        new_baseline = self.baseline / reference if self.has_baseline else None
        ds_new = self._build_ds(self.wavenumber, new_values, tos=self.tos, attrs=dict(self.ds.attrs))
        ds_new = self._with_baseline(ds_new, new_baseline)
        return IRData(ds=self._carry_background(ds_new))

    def normalise_reference_scan(self, scan_index: int) -> "IRData":
        """Normalise all spectra by the scan at ``scan_index``.

        Parameters
        ----------
        scan_index : int
            Zero-based index of the scan to use as the reference.

        Returns
        -------
        IRData
            New instance with each spectrum divided by the reference scan.

        Raises
        ------
        ValueError
            If the data are 1-D.
        """
        if self.ndim == 1:
            raise ValueError("normalise_reference_scan requires 2-D data")
        reference = self.get_scan(scan_index)
        return self.normalise_reference(reference)

    def normalise_reference_by_tos(
        self,
        target_tos: float,
        method: Literal["nearest", "linear"] = "nearest",
        tolerance_seconds: Optional[float] = 10,
    ) -> "IRData":
        """Normalise all spectra by the scan nearest to ``target_tos``.

        Parameters
        ----------
        target_tos : float
            Target elapsed time in seconds.
        method : {'nearest', 'linear'}, optional
            Selection method for the reference scan (default is
            ``'nearest'``).
        tolerance_seconds : float or None, optional
            Maximum distance in seconds between the target and the nearest
            scan (default is ``10``).

        Returns
        -------
        IRData
            New instance with each spectrum divided by the reference scan.

        Raises
        ------
        ValueError
            If the data are 1-D or if the target exceeds the tolerance.
        """
        if self.ndim == 1:
            raise ValueError("normalise_reference_by_tos requires 2-D data")
        reference = self.get_scan_by_tos(target_tos, method=method, tolerance_seconds=tolerance_seconds)
        return self.normalise_reference(reference)

    def normalise_value_range(self, new_min: float = 0.0, new_max: float = 1.0) -> "IRData":
        """Rescale values to fit within ``[new_min, new_max]``.

        Parameters
        ----------
        new_min : float, optional
            Target minimum value (default is ``0.0``).
        new_max : float, optional
            Target maximum value (default is ``1.0``).

        Returns
        -------
        IRData
            New instance with linearly rescaled values.  Returns ``self``
            unchanged if all values are identical.
        """
        old_min = self.values.min()
        old_max = self.values.max()
        if old_max == old_min:
            logger.warning("All values are the same; returning original data without normalisation")
            return self
        scale = (new_max - new_min) / (old_max - old_min)
        new_values = (self.values - old_min) * scale + new_min
        # Baseline is scaled (not shifted) so that data + baseline transforms consistently:
        # the shift applies once to data, while the additive baseline only needs rescaling.
        new_baseline = self.baseline * scale if self.has_baseline else None
        ds_new = self._build_ds(self.wavenumber, new_values, tos=self.tos, attrs=dict(self.ds.attrs))
        ds_new = self._with_baseline(ds_new, new_baseline)
        return IRData(ds=self._carry_background(ds_new))

    def normalise_value(self, factor: float) -> "IRData":
        """Divide all values by a scalar factor.

        Parameters
        ----------
        factor : float
            The divisor.  Must be non-zero.

        Returns
        -------
        IRData
            New instance with values divided by ``factor``.  Returns ``self``
            unchanged if ``factor`` is zero.
        """
        if factor == 0:
            logger.warning("Normalisation factor is zero; returning original data without normalisation")
            return self
        new_values = self.values / factor
        new_baseline = self.baseline / factor if self.has_baseline else None
        ds_new = self._build_ds(self.wavenumber, new_values, tos=self.tos, attrs=dict(self.ds.attrs))
        ds_new = self._with_baseline(ds_new, new_baseline)
        return IRData(ds=self._carry_background(ds_new))

    # ----------------------------------------------------------------
    # Immutable — type conversion
    # ----------------------------------------------------------------

    def _convert(self, new_values: npt.NDArray, new_data_type: IRDataType) -> "IRData":
        self._warn_drop_baseline(
            f"converting data_type to '{new_data_type}' invalidates the baseline curve"
        )
        new_attrs = {**self.ds.attrs, "data_type": new_data_type}
        ds_new = self._build_ds(self.wavenumber, new_values, tos=self.tos, attrs=new_attrs)
        return IRData(ds=self._carry_background(ds_new))

    def _require_background(self) -> npt.NDArray:
        if not self.has_background:
            raise ValueError(
                f"Converting from '{self.data_type}' requires a background spectrum. "
                "Set one with .with_background()."
            )
        return self.background

    def to_transmittance(self) -> "IRData":
        """Convert to transmittance (T = sample / background or T = 10^−A).

        Returns
        -------
        IRData
            New instance with ``data_type='transmittance'``.  Returns ``self``
            if already transmittance.

        Raises
        ------
        ValueError
            If ``data_type`` is a reflectance-based type (``'reflectance'``,
            ``'log_1_r'``, ``'kubelka_munk'``) or if a background is needed
            but not set.
        """
        if self.data_type == "transmittance":
            return self
        if self.data_type == "single_beam":
            bg = self._require_background()
            if np.any(bg == 0):
                logger.warning("Background contains zeros; transmittance will contain inf")
            return self._convert(self.values / bg, "transmittance")
        if self.data_type == "absorbance":
            return self._convert(np.power(10.0, -self.values), "transmittance")
        raise ValueError(
            f"Cannot convert '{self.data_type}' to transmittance. "
            "Reflectance-based types (reflectance, log_1_r, kubelka_munk) belong to a different experiment."
        )

    def to_reflectance(self) -> "IRData":
        """Convert to reflectance (R = sample / background or inverse KM/log(1/R)).

        Returns
        -------
        IRData
            New instance with ``data_type='reflectance'``.  Returns ``self``
            if already reflectance.

        Raises
        ------
        ValueError
            If ``data_type`` is ``'transmittance'`` or if a background is
            needed but not set.
        """
        if self.data_type == "reflectance":
            return self
        if self.data_type == "single_beam":
            bg = self._require_background()
            if np.any(bg == 0):
                logger.warning("Background contains zeros; reflectance will contain inf")
            return self._convert(self.values / bg, "reflectance")
        if self.data_type == "log_1_r":
            return self._convert(np.power(10.0, -self.values), "reflectance")
        if self.data_type == "absorbance":
            # OMNIC stores DRIFTS/reflectance data as absorbance (= -log₁₀(R)); invert to get R.
            return self._convert(np.power(10.0, -self.values), "reflectance")
        if self.data_type == "kubelka_munk":
            km = self.values
            r = (1.0 + km) - np.sqrt(km * (km + 2.0))
            return self._convert(r, "reflectance")
        raise ValueError(
            f"Cannot convert '{self.data_type}' to reflectance. "
            "Transmittance-based type (transmittance) belongs to a different experiment."
        )

    def to_absorbance(self) -> "IRData":
        """Convert to absorbance (A = −log₁₀(T) for transmission experiments).

        Returns
        -------
        IRData
            New instance with ``data_type='absorbance'``.  Returns ``self``
            if already absorbance.

        Raises
        ------
        ValueError
            If ``data_type`` is not ``'single_beam'`` or ``'transmittance'``,
            or if a background is needed but not set.
        """
        if self.data_type == "absorbance":
            return self
        if self.data_type == "single_beam":
            bg = self._require_background()
            ratio = self.values / bg
            if np.any(ratio <= 0):
                logger.warning("Non-positive sample/background ratio; absorbance will contain nan/inf")
            return self._convert(-np.log10(ratio), "absorbance")
        if self.data_type == "transmittance":
            if np.any(self.values <= 0):
                logger.warning("Non-positive transmittance; absorbance will contain nan/inf")
            return self._convert(-np.log10(self.values), "absorbance")
        raise ValueError(
            f"Cannot convert '{self.data_type}' to absorbance. "
            "Absorbance is defined as -log₁₀(T) for transmission experiments. "
            "For reflectance experiments use to_log_1_r()."
        )

    def to_log_1_r(self) -> "IRData":
        """Convert to log(1/R) for diffuse-reflectance experiments.

        Returns
        -------
        IRData
            New instance with ``data_type='log_1_r'``.  Returns ``self``
            if already log(1/R).

        Raises
        ------
        ValueError
            If the current ``data_type`` cannot be converted to log(1/R), or
            if a background is needed but not set.
        """
        if self.data_type == "log_1_r":
            return self
        if self.data_type == "single_beam":
            bg = self._require_background()
            ratio = self.values / bg
            if np.any(ratio <= 0):
                logger.warning("Non-positive sample/background ratio; log(1/R) will contain nan/inf")
            return self._convert(-np.log10(ratio), "log_1_r")
        if self.data_type == "reflectance":
            if np.any(self.values <= 0):
                logger.warning("Non-positive reflectance; log(1/R) will contain nan/inf")
            return self._convert(-np.log10(self.values), "log_1_r")
        if self.data_type == "absorbance":
            # OMNIC stores DRIFTS/reflectance data as absorbance (= -log₁₀(R)); relabel as log(1/R).
            return self._convert(self.values.copy(), "log_1_r")
        if self.data_type == "kubelka_munk":
            km = self.values
            r = (1.0 + km) - np.sqrt(km * (km + 2.0))
            if np.any(r <= 0):
                logger.warning("Non-positive reflectance after KM inversion; log(1/R) will contain nan/inf")
            return self._convert(-np.log10(r), "log_1_r")
        raise ValueError(
            f"Cannot convert '{self.data_type}' to log(1/R). "
            "log(1/R) is defined for reflectance experiments. "
            "For transmission experiments use to_absorbance()."
        )

    def to_kubelka_munk(self) -> "IRData":
        """Convert to Kubelka-Munk remission function F(R) = (1−R)² / (2R).

        Returns
        -------
        IRData
            New instance with ``data_type='kubelka_munk'``.  Returns ``self``
            if already Kubelka-Munk.

        Raises
        ------
        ValueError
            If the current ``data_type`` is transmittance-based and cannot be
            converted to reflectance, or if a background is needed but not
            set.
        """
        if self.data_type == "kubelka_munk":
            return self
        r_ir = self.to_reflectance()  # raises if transmittance-based
        r = r_ir.values
        if np.any(r <= 0):
            logger.warning("Non-positive reflectance; Kubelka-Munk will contain nan/inf")
        return r_ir._convert((1.0 - r) ** 2 / (2.0 * r), "kubelka_munk")

    # ----------------------------------------------------------------
    # Gram-Schmidt reconstructed chromatogram
    # ----------------------------------------------------------------

    def get_gram_schmidt(self, reference: npt.NDArray) -> xr.DataArray:
        """Gram-Schmidt vector: L2 norm of each scan's component orthogonal to the reference subspace.

        Parameters
        ----------
        reference : numpy.ndarray
            1-D or 2-D array of reference spectrum/spectra with shape
            ``(n_wavenumber,)`` or ``(n_ref, n_wavenumber)``.

        Returns
        -------
        xr.DataArray
            1-D DataArray of shape ``(n_scan,)`` containing the Gram-Schmidt
            orthogonal norm for each scan.  Includes ``'tos'`` as a
            non-dimension coordinate when available.

        Raises
        ------
        ValueError
            If the data are 1-D, if the reference wavenumber dimension does
            not match, or if the reference spectra are linearly dependent.
        """
        if self.ndim == 1:
            raise ValueError("get_gram_schmidt requires 2-D data")

        ref = np.atleast_2d(np.asarray(reference, dtype=float))
        if ref.shape[-1] != self.wavenumber.size:
            raise ValueError(
                f"Reference has {ref.shape[-1]} points but wavenumber axis has {self.wavenumber.size}"
            )

        basis: list[npt.NDArray] = []
        for r in ref:
            v = r.copy()
            for b in basis:
                v = v - np.dot(v, b) * b
            norm = np.linalg.norm(v)
            ref_norm = np.linalg.norm(r)
            if norm > 1e-12 * (ref_norm if ref_norm > 0 else 1.0):
                basis.append(v / norm)

        if not basis:
            raise ValueError("Reference spectra are linearly dependent; cannot build orthonormal basis")

        B = np.array(basis)                          # (n_basis, n_wn)
        spectra = self.values                        # (n_scans, n_wn)
        residuals = spectra - (spectra @ B.T) @ B   # remove reference subspace
        gs_values = np.linalg.norm(residuals, axis=1)  # (n_scans,)

        coords: dict[str, Any] = {"scan": self.ds.coords["scan"].values}
        if self.tos is not None:
            coords["tos"] = ("scan", self.tos)

        return xr.DataArray(
            data=gs_values,
            coords=coords,
            dims=["scan"],
            attrs={"gs_n_basis": len(basis)},
            name="gram_schmidt",
        )

    def get_gram_schmidt_scan(
        self,
        reference_scans: Union[int, Sequence[int]] = 0,
    ) -> xr.DataArray:
        """Compute the Gram-Schmidt chromatogram using one or more scans as the reference.

        Parameters
        ----------
        reference_scans : int or sequence of int, optional
            Zero-based index or indices of the scan(s) to use as the reference
            subspace (default is ``0``).

        Returns
        -------
        xr.DataArray
            1-D DataArray of Gram-Schmidt norms, one per scan.

        Raises
        ------
        ValueError
            If the data are 1-D.
        """
        if self.ndim == 1:
            raise ValueError("get_gram_schmidt_scan requires 2-D data")
        indices = [reference_scans] if isinstance(reference_scans, int) else list(reference_scans)
        return self.get_gram_schmidt(self.values[indices])

    def get_gram_schmidt_by_tos(
        self,
        reference_tos: Union[float, Sequence[float]],
        method: Literal["nearest", "linear"] = "nearest",
        tolerance_seconds: Optional[float] = 10,
    ) -> xr.DataArray:
        """Compute the Gram-Schmidt chromatogram using scans selected by tos as the reference.

        Parameters
        ----------
        reference_tos : float or sequence of float
            Target elapsed time(s) in seconds identifying the reference
            scan(s).
        method : {'nearest', 'linear'}, optional
            Selection method (default is ``'nearest'``).
        tolerance_seconds : float or None, optional
            Maximum distance in seconds from the target to the nearest scan
            (default is ``10``).

        Returns
        -------
        xr.DataArray
            1-D DataArray of Gram-Schmidt norms, one per scan.

        Raises
        ------
        ValueError
            If the data are 1-D or if any target exceeds the tolerance.
        """
        if self.ndim == 1:
            raise ValueError("get_gram_schmidt_by_tos requires 2-D data")
        reference = self.get_scan_by_tos(
            reference_tos, method=method, tolerance_seconds=tolerance_seconds
        )
        return self.get_gram_schmidt(reference)

    # ----------------------------------------------------------------
    # SNR
    # ----------------------------------------------------------------

    def _snr_apply(self, fn) -> Union[float, npt.NDArray]:
        if self.ndim == 1:
            return float(fn(self.values))
        return np.array([fn(row) for row in self.values])

    def snr_windows(
        self,
        signal_range_cm: Tuple[float, float],
        noise_range_cm: Tuple[float, float],
        signal_metric: Literal["max", "peak_to_peak", "integral", "rms"] = "max",
        noise_metric: Literal["std", "rms", "peak_to_peak"] = "std",
    ) -> Union[float, npt.NDArray]:
        """Estimate SNR using separate spectral windows for signal and noise.

        Parameters
        ----------
        signal_range_cm : tuple of float
            ``(low, high)`` wavenumber range in cm⁻¹ defining the signal
            window.
        noise_range_cm : tuple of float
            ``(low, high)`` wavenumber range in cm⁻¹ defining the noise
            window.
        signal_metric : {'max', 'peak_to_peak', 'integral', 'rms'}, optional
            Statistic used to quantify the signal (default is ``'max'``).
        noise_metric : {'std', 'rms', 'peak_to_peak'}, optional
            Statistic used to quantify the noise (default is ``'std'``).

        Returns
        -------
        float or numpy.ndarray
            Scalar SNR for 1-D data; array of shape ``(n_scan,)`` for 2-D
            data.

        Raises
        ------
        ValueError
            If no wavenumber points fall within either the signal or noise
            range.
        """
        wn_si = self.wavenumber
        sig_lo, sig_hi = sorted(signal_range_cm)
        noi_lo, noi_hi = sorted(noise_range_cm)
        sig_mask = (wn_si >= sig_lo * 100.0) & (wn_si <= sig_hi * 100.0)
        noi_mask = (wn_si >= noi_lo * 100.0) & (wn_si <= noi_hi * 100.0)
        if not sig_mask.any():
            raise ValueError(f"No points in signal range {sig_lo:.0f}–{sig_hi:.0f} cm⁻¹")
        if not noi_mask.any():
            raise ValueError(f"No points in noise range {noi_lo:.0f}–{noi_hi:.0f} cm⁻¹")

        wn_sig = wn_si[sig_mask]

        def _signal(spec_1d: np.ndarray) -> float:
            s = spec_1d[sig_mask]
            if signal_metric == "max":
                return float(np.abs(s).max())
            if signal_metric == "peak_to_peak":
                return float(s.max() - s.min())
            if signal_metric == "integral":
                return float(np.abs(np.trapz(s, x=wn_sig)))
            return float(np.sqrt(np.mean(s ** 2)))  # rms

        def _noise(spec_1d: np.ndarray) -> float:
            n = spec_1d[noi_mask]
            if noise_metric == "std":
                return float(n.std(ddof=1))
            if noise_metric == "rms":
                return float(np.sqrt(np.mean(n ** 2)))
            return float(n.max() - n.min())  # peak_to_peak

        def _ratio(spec_1d: np.ndarray) -> float:
            sigma = _noise(spec_1d)
            if sigma == 0:
                logger.warning("Noise is zero; returning inf")
                return float("inf")
            return _signal(spec_1d) / sigma

        return self._snr_apply(_ratio)

    def snr_noise_window(
        self,
        noise_range_cm: Tuple[float, float],
        signal_range_cm: Optional[Tuple[float, float]] = None,
        noise_metric: Literal["rms", "std", "peak_to_peak"] = "rms",
        detrend_order: int = 1,
    ) -> Union[float, npt.NDArray]:
        """Estimate SNR using a dedicated noise window with polynomial detrending.

        The noise region is optionally detrended before computing the noise
        metric.  The signal is taken as the absolute maximum over
        ``signal_range_cm`` (or the entire spectrum if ``None``).

        Parameters
        ----------
        noise_range_cm : tuple of float
            ``(low, high)`` wavenumber range in cm⁻¹ used to estimate noise.
        signal_range_cm : tuple of float or None, optional
            ``(low, high)`` wavenumber range in cm⁻¹ used to find the peak
            signal.  ``None`` uses the full spectrum (default is ``None``).
        noise_metric : {'rms', 'std', 'peak_to_peak'}, optional
            Statistic used to quantify the detrended noise (default is
            ``'rms'``).
        detrend_order : int, optional
            Polynomial order for detrending the noise window.  Use ``-1`` to
            skip detrending (default is ``1``).

        Returns
        -------
        float or numpy.ndarray
            Scalar SNR for 1-D data; array of shape ``(n_scan,)`` for 2-D
            data.

        Raises
        ------
        ValueError
            If no points fall within the noise or signal range.
        """
        wn_si = self.wavenumber
        noi_lo, noi_hi = sorted(noise_range_cm)
        noi_mask = (wn_si >= noi_lo * 100.0) & (wn_si <= noi_hi * 100.0)
        if not noi_mask.any():
            raise ValueError(f"No points in noise range {noi_lo:.0f}–{noi_hi:.0f} cm⁻¹")

        if signal_range_cm is not None:
            sig_lo, sig_hi = sorted(signal_range_cm)
            sig_mask = (wn_si >= sig_lo * 100.0) & (wn_si <= sig_hi * 100.0)
            if not sig_mask.any():
                raise ValueError(f"No points in signal range {sig_lo:.0f}–{sig_hi:.0f} cm⁻¹")
        else:
            sig_mask = np.ones_like(wn_si, dtype=bool)

        wn_noi = wn_si[noi_mask]

        def _noise(spec_1d: np.ndarray) -> float:
            n = spec_1d[noi_mask].astype(float)
            if detrend_order >= 0:
                coeffs = np.polyfit(wn_noi, n, deg=detrend_order)
                n = n - np.polyval(coeffs, wn_noi)
            if noise_metric == "rms":
                return float(np.sqrt(np.mean(n ** 2)))
            if noise_metric == "std":
                return float(n.std(ddof=1))
            return float(n.max() - n.min())  # peak_to_peak

        def _ratio(spec_1d: np.ndarray) -> float:
            sigma = _noise(spec_1d)
            if sigma == 0:
                logger.warning("Noise is zero; returning inf")
                return float("inf")
            return float(np.abs(spec_1d[sig_mask]).max()) / sigma

        return self._snr_apply(_ratio)

    def snr_der(
        self,
        signal_range_cm: Optional[Tuple[float, float]] = None,
    ) -> Union[float, npt.NDArray]:
        """Estimate SNR using the DER-SNR algorithm (Stoehr et al. 2008).

        The noise standard deviation is estimated from the second-difference
        of the spectrum without requiring a dedicated noise window.

        Parameters
        ----------
        signal_range_cm : tuple of float or None, optional
            ``(low, high)`` wavenumber range in cm⁻¹ used to find the peak
            signal.  ``None`` uses the full spectrum (default is ``None``).

        Returns
        -------
        float or numpy.ndarray
            Scalar SNR for 1-D data; array of shape ``(n_scan,)`` for 2-D
            data.

        Raises
        ------
        ValueError
            If no points fall within ``signal_range_cm`` or if the spectrum
            has fewer than 5 points.
        """
        wn_si = self.wavenumber
        if signal_range_cm is not None:
            sig_lo, sig_hi = sorted(signal_range_cm)
            sig_mask = (wn_si >= sig_lo * 100.0) & (wn_si <= sig_hi * 100.0)
            if not sig_mask.any():
                raise ValueError(f"No points in signal range {sig_lo:.0f}–{sig_hi:.0f} cm⁻¹")
        else:
            sig_mask = np.ones_like(wn_si, dtype=bool)

        factor = 1.482602 / np.sqrt(6.0)

        def _ratio(spec_1d: np.ndarray) -> float:
            if spec_1d.size < 5:
                raise ValueError("DER-SNR requires at least 5 points")
            diff = 2.0 * spec_1d[2:-2] - spec_1d[:-4] - spec_1d[4:]
            sigma = factor * float(np.median(np.abs(diff)))
            if sigma == 0:
                logger.warning("Noise is zero; returning inf")
                return float("inf")
            return float(np.abs(spec_1d[sig_mask]).max()) / sigma

        return self._snr_apply(_ratio)

    def snr_repeat(
        self,
        signal_range_cm: Optional[Tuple[float, float]] = None,
        reduce: Literal["max", "median", "mean", "per_wavenumber"] = "max",
    ) -> Union[float, npt.NDArray]:
        """Estimate SNR from scan-to-scan reproducibility across repeated measurements.

        The noise is the standard deviation across scans; the signal is the
        mean absolute value.

        Parameters
        ----------
        signal_range_cm : tuple of float or None, optional
            ``(low, high)`` wavenumber range in cm⁻¹ used to find the peak
            SNR.  ``None`` uses the full spectrum (default is ``None``).
        reduce : {'max', 'median', 'mean', 'per_wavenumber'}, optional
            How to summarise the per-wavenumber SNR across the signal range
            (default is ``'max'``).  Use ``'per_wavenumber'`` to return the
            full SNR spectrum.

        Returns
        -------
        float or numpy.ndarray
            Scalar SNR when ``reduce`` is not ``'per_wavenumber'``; array of
            shape ``(n_wavenumber,)`` when ``reduce='per_wavenumber'``.

        Raises
        ------
        ValueError
            If the data are 1-D, if fewer than 2 scans are present, or if no
            points fall within ``signal_range_cm``.
        """
        if self.ndim == 1:
            raise ValueError("snr_repeat requires 2-D data")
        if self.shape[0] < 2:
            raise ValueError("snr_repeat requires at least 2 scans")

        wn_si = self.wavenumber
        if signal_range_cm is not None:
            sig_lo, sig_hi = sorted(signal_range_cm)
            sig_mask = (wn_si >= sig_lo * 100.0) & (wn_si <= sig_hi * 100.0)
            if not sig_mask.any():
                raise ValueError(f"No points in signal range {sig_lo:.0f}–{sig_hi:.0f} cm⁻¹")
        else:
            sig_mask = np.ones_like(wn_si, dtype=bool)

        mean_spec = self.values.mean(axis=0)
        sigma_spec = self.values.std(axis=0, ddof=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            snr_per_wn = np.where(sigma_spec > 0, np.abs(mean_spec) / sigma_spec, np.inf)

        if reduce == "per_wavenumber":
            return snr_per_wn

        snr_in_range = snr_per_wn[sig_mask]
        if reduce == "max":
            return float(np.nanmax(snr_in_range))
        if reduce == "median":
            return float(np.nanmedian(snr_in_range))
        return float(np.nanmean(snr_in_range))  # mean

    def snr_psd(
        self,
        signal_range_cm: Optional[Tuple[float, float]] = None,
        noise_fraction: float = 0.25,
        detrend_order: int = 1,
    ) -> Union[float, npt.NDArray]:
        """Estimate SNR from the high-frequency tail of the power spectral density.

        The noise power is estimated from the top ``noise_fraction`` of FFT
        frequency bins.

        Parameters
        ----------
        signal_range_cm : tuple of float or None, optional
            ``(low, high)`` wavenumber range in cm⁻¹ used to find the peak
            signal.  ``None`` uses the full spectrum (default is ``None``).
        noise_fraction : float, optional
            Fraction of high-frequency FFT bins treated as noise, in
            ``(0, 1)`` (default is ``0.25``).
        detrend_order : int, optional
            Polynomial order for detrending before the FFT.  Use ``-1`` to
            skip (default is ``1``).

        Returns
        -------
        float or numpy.ndarray
            Scalar SNR for 1-D data; array of shape ``(n_scan,)`` for 2-D
            data.

        Raises
        ------
        ValueError
            If ``noise_fraction`` is not in ``(0, 1)``, if the spectrum has
            fewer than 8 points, or if no points fall within
            ``signal_range_cm``.
        """
        if not (0.0 < noise_fraction < 1.0):
            raise ValueError("noise_fraction must be in (0, 1)")

        wn_si = self.wavenumber
        n_pts = wn_si.size
        if n_pts < 8:
            raise ValueError("snr_psd requires at least 8 points")

        if signal_range_cm is not None:
            sig_lo, sig_hi = sorted(signal_range_cm)
            sig_mask = (wn_si >= sig_lo * 100.0) & (wn_si <= sig_hi * 100.0)
            if not sig_mask.any():
                raise ValueError(f"No points in signal range {sig_lo:.0f}–{sig_hi:.0f} cm⁻¹")
        else:
            sig_mask = np.ones_like(wn_si, dtype=bool)

        # Number of high-frequency FFT bins to treat as noise
        n_freq = n_pts // 2 + 1
        n_noise_bins = max(1, int(np.floor(n_freq * noise_fraction)))

        x_axis = np.arange(n_pts, dtype=float)

        def _noise(spec_1d: np.ndarray) -> float:
            s = spec_1d.astype(float)
            if detrend_order >= 0:
                coeffs = np.polyfit(x_axis, s, deg=detrend_order)
                s = s - np.polyval(coeffs, x_axis)
            spectrum = np.fft.rfft(s)
            psd = (np.abs(spectrum) ** 2) / n_pts
            tail = psd[-n_noise_bins:]
            # Parseval: noise variance ≈ mean PSD over the tail bins
            return float(np.sqrt(tail.mean()))

        def _ratio(spec_1d: np.ndarray) -> float:
            sigma = _noise(spec_1d)
            if sigma == 0:
                logger.warning("Noise is zero; returning inf")
                return float("inf")
            return float(np.abs(spec_1d[sig_mask]).max()) / sigma

        return self._snr_apply(_ratio)

    # ----------------------------------------------------------------
    # Export
    # ----------------------------------------------------------------

    def to_netcdf(self, filepath: Union[str, Path]) -> None:
        """Save the dataset to a NetCDF file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Destination file path.  An existing file will be overwritten with
            a warning.
        """
        # tos_start in ds.attrs round-trips automatically
        filepath = Path(filepath)
        if filepath.exists():
            logger.warning(f"Overwriting existing file: {filepath}")
        self.ds.to_netcdf(filepath)
        logger.debug(f"Saved NetCDF → {filepath}")


    # ----------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        wavenumber_per_cm: npt.NDArray,
        values: npt.NDArray,
        tos: Optional[npt.NDArray] = None,
        tos_start: Optional[Union[pd.Timestamp, str]] = None,
        data_type: IRDataType = "single_beam",
    ) -> "IRData":
        """Construct an IRData from raw NumPy arrays.

        Parameters
        ----------
        wavenumber_per_cm : numpy.ndarray
            1-D wavenumber axis in cm⁻¹.
        values : numpy.ndarray
            Spectral values.  Shape ``(n_wavenumber,)`` for a single spectrum
            or ``(n_scan, n_wavenumber)`` for a time series.
        tos : numpy.ndarray or None, optional
            1-D array of elapsed times in seconds, required for 2-D data
            (default is ``None``).
        tos_start : pandas.Timestamp or str or None, optional
            Absolute start time of the measurement (default is ``None``).
        data_type : str, optional
            Spectral quantity label (default is ``'single_beam'``).

        Returns
        -------
        IRData
            New instance built from the provided arrays.

        Raises
        ------
        ValueError
            If ``wavenumber_per_cm`` is not 1-D, if ``values`` shape is
            inconsistent with the wavenumber axis, or if ``tos`` size does not
            match the number of scans.
        """
        wavenumber_si = np.asarray(wavenumber_per_cm, dtype=float) * 100.0
        values = np.asarray(values, dtype=float)

        if wavenumber_si.ndim != 1:
            raise ValueError("wavenumber_per_cm must be 1-D")

        if values.ndim == 1:
            if values.size != wavenumber_si.size:
                raise ValueError(f"values size ({values.size}) != wavenumber size ({wavenumber_si.size})")
        elif values.ndim == 2:
            n_scans, n_pts = values.shape
            if n_pts != wavenumber_si.size:
                raise ValueError(f"values.shape[1] ({n_pts}) != wavenumber size ({wavenumber_si.size})")
            if tos is not None:
                tos = np.asarray(tos, dtype=float)
                if tos.ndim != 1 or tos.size != n_scans:
                    raise ValueError(f"tos size ({tos.size}) != values.shape[0] ({n_scans})")
        else:
            raise ValueError(f"values must be 1-D or 2-D, got shape {values.shape}")

        attrs: dict[str, Any] = {"data_type": data_type}
        if tos_start is not None:
            attrs["tos_start"] = pd.Timestamp(tos_start).isoformat()

        ds = cls._build_ds(wavenumber_si, values, tos=tos, attrs=attrs)
        return cls(ds=ds)

    @classmethod
    def from_netcdf(cls, filepath: Union[str, Path]) -> "IRData":
        """Load an IRData from a NetCDF file previously saved with ``to_netcdf``.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to the ``.nc`` file.

        Returns
        -------
        IRData
            New instance reconstructed from the file.
        """
        with xr.open_dataset(filepath) as ds:
            ds = ds.copy()
        return cls(ds=ds)

    @classmethod
    def from_xarray(
        cls,
        da: Union[xr.DataArray, xr.Dataset],
    ) -> "IRData":
        """Construct an IRData from an existing xarray DataArray or Dataset.

        Parameters
        ----------
        da : xr.DataArray or xr.Dataset
            Source object.  A ``DataArray`` is wrapped in a ``Dataset`` under
            the key ``'data'``.  A ``Dataset`` is used directly.

        Returns
        -------
        IRData
            New instance wrapping the provided xarray object.
        """
        if isinstance(da, xr.DataArray):
            ds = xr.Dataset({"data": da.copy()}, attrs=dict(da.attrs))
        else:
            ds = da.copy()
        return cls(ds=ds)

    @classmethod
    def from_omnic_spa(
        cls,
        filepath: Union[str, Path],
        wavenumber_2SI_factor: float = 100.0,
        delta_time_seconds: Optional[float] = None,
        tos_start: Optional[Union[pd.Timestamp, str]] = None,
        strict_tos_start: bool = True,
        backend: Literal["auto", "spectrochempy", "omnic"] = "auto",
    ) -> "IRData":
        """Load one or more Thermo OMNIC ``.spa`` files into an IRData.

        A single file path produces a 1-D instance; a directory path (or the
        ``filepath`` pointing to a directory) reads all ``.spa`` files in that
        directory and produces a 2-D instance sorted by spectrum index.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to a single ``.spa`` file or a directory containing multiple
            ``.spa`` files.
        wavenumber_2SI_factor : float, optional
            Multiplicative factor to convert the file's native x-axis to m⁻¹
            (default is ``100.0``, converting cm⁻¹ → m⁻¹).
        delta_time_seconds : float or None, optional
            Fixed time step between consecutive spectra used to build the
            ``tos`` coordinate when embedded timestamps are unavailable.
            Mutually exclusive with ``tos_start`` (default is ``None``).
        tos_start : pandas.Timestamp or str or None, optional
            Absolute start time used to compute ``tos`` from file timestamps.
            Mutually exclusive with ``delta_time_seconds`` (default is
            ``None``).
        strict_tos_start : bool, optional
            Raise ``ValueError`` if the ``tos_start`` from file metadata
            cannot be parsed when ``True``; log a warning and skip it when
            ``False`` (default is ``True``).
        backend : {"auto", "spectrochempy", "omnic"}, optional
            Low-level file reader to use.  ``"auto"`` (default) tries
            spectrochempy first and falls back to the built-in omnic parser
            when spectrochempy is not installed.  ``"spectrochempy"`` forces
            the spectrochempy reader (raises ``ImportError`` if not available).
            ``"omnic"`` forces the built-in binary parser.

        Returns
        -------
        IRData
            New instance loaded from the file(s).

        Raises
        ------
        ValueError
            If both ``delta_time_seconds`` and ``tos_start`` are provided, if
            the OMNIC y-axis label is not in the known mapping, or if
            ``strict_tos_start`` is ``True`` and ``tos_start`` cannot be
            parsed.
        ImportError
            If ``backend="spectrochempy"`` and spectrochempy is not installed.
        """
        if delta_time_seconds is not None and tos_start is not None:
            raise ValueError("Specify either 'delta_time_seconds' or 'tos_start', not both.")

        from phd_parser.infrared import spectrochempy as _scp_parser

        if backend == "spectrochempy":
            raw = _scp_parser.read_spa(
                filepath, delta_time_seconds=delta_time_seconds, tos_start=tos_start
            )
        elif backend == "omnic":
            raw = omnic.read_spa(filepath, delta_time_seconds=delta_time_seconds, tos_start=tos_start)
        else:  # "auto"
            try:
                raw = _scp_parser.read_spa(
                    filepath, delta_time_seconds=delta_time_seconds, tos_start=tos_start
                )
                logger.debug("from_omnic_spa: using spectrochempy backend")
            except ImportError:
                logger.debug("spectrochempy not available; falling back to built-in omnic parser")
                raw = omnic.read_spa(
                    filepath, delta_time_seconds=delta_time_seconds, tos_start=tos_start
                )

        wavenumber_si = np.asarray(raw["data"]["x"]) * wavenumber_2SI_factor
        values = np.asarray(raw["data"]["v"], dtype=float)
        tos = np.asarray(raw["data"].get("tos"), dtype=float) if "tos" in raw["data"] else None

        # Parse tos_start from argument or attributes, with optional strictness
        parsed_tos_start: Optional[pd.Timestamp] = None
        if tos_start is not None:
            parsed_tos_start = pd.Timestamp(tos_start)
        elif (raw_ts := raw["meta"].get("tos_start")) is not None:
            try:
                parsed_tos_start = pd.Timestamp(raw_ts)
            except Exception as exc:
                if strict_tos_start:
                    raise ValueError(f"Could not parse tos_start '{raw_ts}': {exc}") from exc
                logger.warning(f"Ignoring unparseable tos_start '{raw_ts}': {exc}")

        attrs = {}
        if parsed_tos_start is not None:
            attrs["tos_start"] = parsed_tos_start.isoformat()

        vlabel = raw["meta"].get("vlabel", "")
        data_type = _OMNIC_VLABEL_TO_DATA_TYPE.get(vlabel)
        if data_type is None:
            raise ValueError(
                f"Omnic vlabel '{vlabel}' has no known data_type mapping. "
                f"Known mappings: {list(_OMNIC_VLABEL_TO_DATA_TYPE)}. "
                "Add it to _OMNIC_VLABEL_TO_DATA_TYPE in infrared/core.py."
            )
        attrs["data_type"] = data_type

        ds = cls._build_ds(
            wavenumber_si,
            values,
            tos=tos,
            attrs=attrs,
        )

        return cls(ds=ds)

    # ----------------------------------------------------------------
    # Dunder helpers
    # ----------------------------------------------------------------

    def __repr__(self) -> str:
        wn = self.wavenumber_per_cm
        wn_range = f"{wn.min():.1f}–{wn.max():.1f} cm-1" if wn.size else "empty"
        tos_info = f", tos={self.tos[0]:.1f}–{self.tos[-1]:.1f}s" if self.tos is not None else ""
        ts_info = f", tos_start={self.tos_start}" if self.tos_start is not None else ""
        bg_info = f", background={self.background_data_type}" if self.has_background else ""
        bl_info = ", baseline=stored" if self.has_baseline else ""
        return f"IRData(data_type={self.data_type}, shape={self.shape}, wavenumber={wn_range}{tos_info}{ts_info}{bg_info}{bl_info})"

    def __len__(self) -> int:
        return self.ds.sizes.get("scan", 1)

    def __add__(self, other: "IRData") -> "IRData":
        if not isinstance(other, IRData):
            return NotImplemented
        self._check_compatible(other, "add")
        self._warn_drop_baseline("combining two IRData via __add__ invalidates either baseline")
        other._warn_drop_baseline("combining two IRData via __add__ invalidates either baseline")
        new_attrs = {**self.ds.attrs, **other.ds.attrs}
        ds_new = self._build_ds(self.wavenumber, self.values + other.values, tos=self.tos, attrs=new_attrs)
        return IRData(ds=self._carry_background(ds_new))

    def __sub__(self, other: "IRData") -> "IRData":
        if not isinstance(other, IRData):
            return NotImplemented
        self._check_compatible(other, "subtract")
        self._warn_drop_baseline("combining two IRData via __sub__ invalidates either baseline")
        other._warn_drop_baseline("combining two IRData via __sub__ invalidates either baseline")
        new_attrs = {**self.ds.attrs, **other.ds.attrs}
        ds_new = self._build_ds(self.wavenumber, self.values - other.values, tos=self.tos, attrs=new_attrs)
        return IRData(ds=self._carry_background(ds_new))


    # ----------------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------------

    def _del_background(self) -> "IRData":
        if "background" not in self.ds:
            return self
        return IRData(ds=self.ds.drop_vars("background"))

    def _make_bg_da(self, values: npt.NDArray) -> xr.DataArray:
        return xr.DataArray(
            data=values,
            coords={"wavenumber": self.ds.coords["wavenumber"]},
            dims=["wavenumber"],
            name="background",
            attrs={"data_type": "single_beam"},
        )

    def _set_background(self, bg_single_beam: npt.NDArray) -> "IRData":
        return IRData(ds=self.ds.assign({"background": self._make_bg_da(bg_single_beam)}))

    def _switch_background(self, new_bg: npt.NDArray) -> "IRData":
        self._warn_drop_baseline("switching background recalculates data values")
        new_values = self._recalculate_with_new_background(new_bg)
        new_attrs = dict(self.ds.attrs)
        ds_new = self._build_ds(self.wavenumber, new_values, tos=self.tos, attrs=new_attrs)
        return IRData(ds=ds_new.assign({"background": self._make_bg_da(new_bg)}))

    def _bg_to_single_beam(
        self, bg_values: npt.NDArray, bg_data_type: Optional[IRDataType]
    ) -> npt.NDArray:
        """Convert incoming background values to single_beam units.

        If bg_data_type is already single_beam (or None), return as-is.
        Otherwise reconstruct using the existing background (which must exist).
        """
        if bg_data_type is None or bg_data_type == "single_beam":
            return bg_values
        old_bg = self.background
        if old_bg is None:
            raise ValueError(
                f"Cannot convert background from '{bg_data_type}' to single_beam without an "
                "existing background. Assign a single_beam background first, or pass "
                "data_type='single_beam'."
            )
        if bg_data_type in ("transmittance", "reflectance"):
            return bg_values * old_bg
        if bg_data_type in ("absorbance", "log_1_r"):
            return np.power(10.0, -bg_values) * old_bg
        if bg_data_type == "kubelka_munk":
            km = bg_values
            r = (1.0 + km) - np.sqrt(km * (km + 2.0))
            return r * old_bg
        raise ValueError(f"Unknown background data_type '{bg_data_type}'")

    def _recalculate_with_new_background(self, new_bg: npt.NDArray) -> npt.NDArray:
        """Recalculate data values after swapping background (old background must exist)."""
        old_bg = self.background
        dt = self.data_type
        if dt is None:
            raise ValueError("data_type is not set; cannot recalculate values with new background")
        if dt == "single_beam":
            return self.values.copy()
        if dt in ("transmittance", "reflectance"):
            return self.values * (old_bg / new_bg)
        if dt in ("absorbance", "log_1_r"):
            return self.values + np.log10(new_bg / old_bg)
        if dt == "kubelka_munk":
            km = self.values
            r_old = (1.0 + km) - np.sqrt(km * (km + 2.0))
            r_new = r_old * (old_bg / new_bg)
            return (1.0 - r_new) ** 2 / (2.0 * r_new)
        raise ValueError(f"Cannot recalculate values for data_type '{dt}'")

    def _carry_background(self, ds: xr.Dataset) -> xr.Dataset:
        """Copy background variable from self into ds unchanged (wavenumber axis must match)."""
        if "background" in self.ds:
            return ds.assign({"background": self.ds["background"]})
        return ds

    def _slice_background_to(self, ds: xr.Dataset) -> xr.Dataset:
        """Slice background to match ds's wavenumber coordinate, then copy into ds."""
        if "background" not in self.ds:
            return ds
        bg_sliced = self.ds["background"].sel(wavenumber=ds.coords["wavenumber"])
        return ds.assign({"background": bg_sliced})

    def _del_baseline(self) -> "IRData":
        if "baseline" not in self.ds:
            return self
        return IRData(ds=self.ds.drop_vars("baseline"))

    def _accumulate_baseline(self, increment: npt.NDArray) -> npt.NDArray:
        """Add increment to the existing stored baseline, or return it as the new baseline."""
        if self.has_baseline:
            return self.baseline + increment
        return increment

    def _with_baseline(self, ds: xr.Dataset, baseline_values: Optional[npt.NDArray]) -> xr.Dataset:
        """Attach baseline_values to ds, shaped/coordinated like ds['data']."""
        if baseline_values is None:
            return ds
        da = xr.DataArray(
            data=baseline_values, coords=ds["data"].coords, dims=ds["data"].dims, name="baseline"
        )
        return ds.assign({"baseline": da})

    def _carry_baseline(self, ds: xr.Dataset) -> xr.Dataset:
        """Copy baseline variable from self into ds unchanged (dims/shape must match)."""
        if "baseline" in self.ds:
            return ds.assign({"baseline": self.ds["baseline"]})
        return ds

    def _slice_baseline_to(self, ds: xr.Dataset) -> xr.Dataset:
        """Slice baseline to match ds's wavenumber coordinate, then copy into ds."""
        if "baseline" not in self.ds:
            return ds
        bl_sliced = self.ds["baseline"].sel(wavenumber=ds.coords["wavenumber"])
        return ds.assign({"baseline": bl_sliced})

    def _warn_drop_baseline(self, reason: str) -> None:
        if self.has_baseline:
            logger.warning(
                f"Dropping stored baseline: {reason}. Re-run baseline correction afterward if needed."
            )

    def _select_var_by_tos(
        self,
        var: xr.DataArray,
        target_tos: Union[float, Sequence[float]],
        method: Literal["nearest", "linear"],
        tolerance_seconds: Optional[float],
    ) -> npt.NDArray:
        """Select scan(s) of var nearest to one or more target tos values (shared by get_scan_by_tos/get_baseline_by_tos)."""
        scalar_input = np.ndim(target_tos) == 0
        targets = [float(target_tos)] if scalar_input else [float(t) for t in target_tos]

        def _fetch_one(t: float) -> npt.NDArray:
            if tolerance_seconds is not None:
                nearest_dist = float(np.abs(self.tos - t).min())
                if nearest_dist > tolerance_seconds:
                    raise ValueError(
                        f"Requested tos {t:.1f}s is {nearest_dist:.1f}s from the nearest scan "
                        f"(tolerance: {tolerance_seconds:.1f}s)"
                    )
            return var.sel(tos=t, method=method).values

        results = np.vstack([_fetch_one(t) for t in targets])
        return results[0] if scalar_input else results

    def _check_compatible(self, other: "IRData", op: str) -> None:
        if self.wavenumber.shape != other.wavenumber.shape or not np.allclose(self.wavenumber, other.wavenumber):
            raise ValueError(f"Cannot {op} IRData with different wavenumber axes")
        if self.ndim != other.ndim:
            raise ValueError(f"Cannot {op} IRData with different number of dimensions")
        if self.ndim == 2 and self.shape[0] != other.shape[0]:
            raise ValueError(f"Cannot {op} 2-D IRData with different number of scans")


    @staticmethod
    def _build_ds(
        wavenumber_si: npt.NDArray,
        values: npt.NDArray,
        tos: Optional[npt.NDArray] = None,
        attrs: Optional[dict[str, Any]] = None,
    ) -> xr.Dataset:
        coords: dict[str, Any] = {"wavenumber": wavenumber_si}
        dims: list[str]

        if attrs is None:
            attrs = {}

        attrs["wavenumber_unit"] = "m^-1"
        attrs.setdefault("data_type", "single_beam")

        if values.ndim == 1:
            dims = ["wavenumber"]
        else:
            dims = ["scan", "wavenumber"]
            coords["scan"] = np.arange(values.shape[0])
            if tos is not None:
                coords["tos"] = ("scan", np.asarray(tos, dtype=float))
                attrs["tos_unit"] = "s"
            else:
                logger.warning("No 'tos' provided for 2-D data; 'tos' coordinate will be missing")

        data_var = xr.DataArray(data=values, coords=coords, dims=dims, name="data")
        ds = xr.Dataset({"data": data_var}, attrs=attrs)

        logger.debug(f"Built Dataset with dims={dict(ds.sizes)}, coords={list(ds.coords)}, shape={ds['data'].shape}, tos_start={attrs.get('tos_start') if attrs else None}, tos[0]={tos[0] if tos is not None else None}, tos[-1]={tos[-1] if tos is not None else None}, attribute_keys={list(attrs.keys()) if attrs else None}")
        return ds
