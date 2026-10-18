"""X-ray photoelectron spectra: :class:`XPSData`, one region (one scan window) of one sample.

An XPS measurement is a set of regions (``XPS Survey``, ``C1s Scan``,
``O1s Scan``, ...), each a photoelectron count over a kinetic energy window
of the analyser. The instrument scans kinetic energy; what everyone plots is
the binding energy, ``E_B = hν - E_K`` with the source energy ``hν`` (Thermo's
convention, the analyser work function is kept in the metadata and not
subtracted). ``XPSData`` stores one region as an ``xarray.Dataset`` with the
``data`` variable over ``binding_energy`` in eV, the source energy in the
attributes so the kinetic energy is always recoverable, and whatever else
the file carried (the analyser transmission, a CasaXPS fit) as further
variables. A file with several regions reads into a ``dict`` of them, like
the two branches of a physisorption isotherm.

Readers: :meth:`XPSData.from_vamas` (ISO 14976 ``.vms``, the raw export of
Avantage and what CasaXPS reads), :meth:`XPSData.from_avantage_avg` (Thermo
Avantage ``.avg`` text dump) and :meth:`XPSData.from_casaxps_export` (the
tab-separated export of a CasaXPS fit). :meth:`XPSData.fit_peaks` does the
CasaXPS curve fit here (``xps/fitting.py``) and stores it the same way.
Every processing step returns a new object and is recorded in the history.
"""

import logging
from collections.abc import Sequence
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, field_validator

from phd_parser.history import (
    HistoryMixin,
    decode_attrs,
    encode_attrs,
    records_step,
    to_json_safe,
)

logger = logging.getLogger(__name__)

DataType = Literal["counts", "cps", "arbitrary"]
AXIS = "binding_energy"


def _kinetic_to_binding(kinetic_energy_eV: npt.NDArray, source_energy_eV: float) -> npt.NDArray:
    return float(source_energy_eV) - np.asarray(kinetic_energy_eV, dtype=float)


class XPSData(HistoryMixin, BaseModel):
    """Immutable wrapper around one XPS region: counts over binding energy, with the acquisition parameters.

    Attributes
    ----------
    ds : xr.Dataset
        ``data`` over the ``binding_energy`` dimension (eV, ascending);
        optional 1-D variables ``transmission`` (the analyser transmission
        function, VAMAS), ``background`` (a subtracted or fitted background),
        ``envelope`` (the sum of a fit) and the 2-D ``peaks`` over
        ``(peak, binding_energy)`` with the coordinates ``peak_name``,
        ``peak_position_eV``, ``peak_fwhm_eV``, ``peak_area`` and
        ``peak_lineshape`` of a fit, imported from CasaXPS or made by
        :meth:`fit_peaks` (which adds ``peak_area_fraction`` and the
        summary ``ds.attrs['fit']``). The peak curves and the envelope sit on
        top of the ``background``, as CasaXPS exports them. ``ds.attrs`` hold
        ``technique`` (``'xps'``), ``region``, ``sample``, ``species_label``,
        ``source_energy_eV``, ``data_type`` (``'counts'``, ``'cps'`` or
        ``'arbitrary'``), ``dwell_time_s``, ``n_scans``, ``pass_energy_eV``,
        ``acquisition_time`` (ISO string), ``binding_energy_shift_eV`` (the
        cumulative charge correction applied), ``analysis_source`` and the
        ``history``.
    metadata : dict
        Everything else the reader found in the file, JSON-safe.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True, ignored_types=(cached_property,))

    ds: xr.Dataset = Field(description="Dataset with 'data' over 'binding_energy' (eV).")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Raw metadata of the source file.")

    @field_validator("ds", mode="before")
    @classmethod
    def validate_ds(cls, v: Any) -> xr.Dataset:
        """Check the layout of ``ds``: a 1-D ``data`` variable over ``binding_energy``."""
        if not isinstance(v, xr.Dataset):
            raise TypeError(f"'ds' must be an xr.Dataset, got {type(v)}")
        if "data" not in v:
            raise ValueError("Dataset must contain a 'data' variable")
        if tuple(v["data"].dims) != (AXIS,):
            raise ValueError(f"'data' must be 1-D over '{AXIS}', got dims {v['data'].dims}")
        for name in ("transmission", "background", "envelope"):
            if name in v and tuple(v[name].dims) != (AXIS,):
                raise ValueError(f"'{name}' must be 1-D over '{AXIS}'")
        if "peaks" in v and tuple(v["peaks"].dims) != ("peak", AXIS):
            raise ValueError("'peaks' must have dims ('peak', 'binding_energy')")
        return v

    # ----------------------------------------------------------------
    # Accessors
    # ----------------------------------------------------------------

    @property
    def binding_energy(self) -> npt.NDArray:
        """Binding energy axis in eV."""
        return self.ds.coords[AXIS].values

    @property
    def kinetic_energy(self) -> npt.NDArray:
        """Kinetic energy axis in eV, ``source_energy_eV - binding_energy``.

        Raises
        ------
        ValueError
            If no source energy is stored.
        """
        if self.source_energy_eV is None:
            raise ValueError("No source_energy_eV stored; the kinetic energy cannot be computed.")
        return self.source_energy_eV - self.binding_energy

    @property
    def values(self) -> npt.NDArray:
        """The spectrum, in the units of :attr:`data_type`."""
        return self.ds["data"].values

    @property
    def n_points(self) -> int:
        """Number of points on the energy axis."""
        return self.ds.sizes[AXIS]

    @property
    def data_type(self) -> Optional[str]:
        """What the values are: ``'counts'``, ``'cps'`` or ``'arbitrary'``."""
        return self.ds.attrs.get("data_type")

    @property
    def region(self) -> Optional[str]:
        """Name of the region as the instrument called it (``'C1s Scan'``, ``'XPS Survey'``)."""
        return self.ds.attrs.get("region")

    @property
    def sample(self) -> Optional[str]:
        """Sample identifier from the file."""
        return self.ds.attrs.get("sample")

    @property
    def species_label(self) -> Optional[str]:
        """The core level the region was scanned for (``'C 1s'``), as the file labels it."""
        return self.ds.attrs.get("species_label")

    @property
    def source_energy_eV(self) -> Optional[float]:
        """Photon energy of the X-ray source in eV (Al Kα: 1486.68)."""
        value = self.ds.attrs.get("source_energy_eV")
        return None if value is None else float(value)

    @property
    def dwell_time_s(self) -> Optional[float]:
        """Dwell time per point in s."""
        value = self.ds.attrs.get("dwell_time_s")
        return None if value is None else float(value)

    @property
    def n_scans(self) -> Optional[int]:
        """Number of scans summed into the region."""
        value = self.ds.attrs.get("n_scans")
        return None if value is None else int(value)

    @property
    def acquisition_time(self) -> Optional[pd.Timestamp]:
        """When the region was recorded, from the file."""
        raw = self.ds.attrs.get("acquisition_time")
        return None if raw is None else pd.Timestamp(raw)

    @property
    def has_background(self) -> bool:
        """Whether a background curve is stored (subtracted here, or fitted by CasaXPS)."""
        return "background" in self.ds

    @property
    def background(self) -> Optional[npt.NDArray]:
        """The stored background curve, or ``None``."""
        return self.ds["background"].values if "background" in self.ds else None

    @property
    def transmission(self) -> Optional[npt.NDArray]:
        """The analyser transmission function along the axis (VAMAS files), or ``None``."""
        return self.ds["transmission"].values if "transmission" in self.ds else None

    @property
    def has_fit(self) -> bool:
        """Whether a peak fit (CasaXPS) is stored."""
        return "peaks" in self.ds

    @property
    def peaks(self) -> Optional[pd.DataFrame]:
        """The fitted peaks as a table (name, position, FWHM, area, lineshape), or ``None``."""
        if "peaks" not in self.ds:
            return None
        return pd.DataFrame({
            "name": self.ds.coords["peak_name"].values,
            "position_eV": self.ds.coords["peak_position_eV"].values,
            "fwhm_eV": self.ds.coords["peak_fwhm_eV"].values,
            "area": self.ds.coords["peak_area"].values,
            "lineshape": self.ds.coords["peak_lineshape"].values,
        })

    @property
    def envelope(self) -> Optional[npt.NDArray]:
        """The fit envelope (sum of peaks and background), or ``None``."""
        return self.ds["envelope"].values if "envelope" in self.ds else None

    def peak_table(self) -> pd.DataFrame:
        """The fitted peaks as one row each, for a CasaXPS import and for :meth:`fit_peaks` alike.

        Returns
        -------
        pandas.DataFrame
            Columns ``name``, ``position_eV``, ``fwhm_eV``, ``area``,
            ``area_fraction`` (of the summed area), ``height`` (maximum of
            the peak curve above the background) and ``lineshape``; a fit
            made here adds ``position_error_eV``, ``fwhm_error_eV`` and
            ``area_error``.

        Raises
        ------
        ValueError
            If no fit is stored.
        """
        if not self.has_fit:
            raise ValueError("No peak fit stored; run fit_peaks or read a CasaXPS export")
        coords = self.ds.coords
        area = np.asarray(coords["peak_area"].values, dtype=float)
        if "peak_area_fraction" in coords:
            fraction = np.asarray(coords["peak_area_fraction"].values, dtype=float)
        else:
            total = float(np.nansum(area))
            fraction = area / total if total else np.full_like(area, np.nan)
        curves = self.ds["peaks"].values
        offset = self.background if self.has_background else 0.0
        frame = pd.DataFrame({
            "name": coords["peak_name"].values,
            "position_eV": np.asarray(coords["peak_position_eV"].values, dtype=float),
            "fwhm_eV": np.asarray(coords["peak_fwhm_eV"].values, dtype=float),
            "area": area,
            "area_fraction": fraction,
            "height": np.nanmax(curves - offset, axis=1),
            "lineshape": coords["peak_lineshape"].values,
        })
        fitted = self.ds.attrs.get("fit", {}).get("peaks") if isinstance(self.ds.attrs.get("fit"), dict) else None
        if fitted and [entry.get("name") for entry in fitted] == list(frame["name"]):
            for column in ("position_error_eV", "fwhm_error_eV", "area_error"):
                frame[column] = [np.nan if entry.get(column) is None else float(entry[column]) for entry in fitted]
        return frame

    def integrate(self, min_eV: Optional[float] = None, max_eV: Optional[float] = None) -> float:
        """Area under the spectrum over a binding energy window (trapezoid rule).

        Parameters
        ----------
        min_eV, max_eV : float or None, optional
            Inclusive bounds in eV; ``None`` leaves that side open.

        Returns
        -------
        float
            The area in ``data_type`` units times eV.
        """
        energy = self.binding_energy
        mask = np.ones(energy.size, dtype=bool)
        if min_eV is not None:
            mask &= energy >= min_eV
        if max_eV is not None:
            mask &= energy <= max_eV
        if mask.sum() < 2:
            raise ValueError(f"Fewer than two points in [{min_eV}, {max_eV}] eV")
        return float(np.trapezoid(self.values[mask], x=energy[mask]))

    # ----------------------------------------------------------------
    # Immutable processing
    # ----------------------------------------------------------------

    def _new(self, ds: xr.Dataset) -> "XPSData":
        return XPSData(ds=ds, metadata=self.metadata)

    def _with_values(self, values: npt.NDArray, **attrs: Any) -> "XPSData":
        ds = self.ds.copy()
        ds["data"] = xr.DataArray(np.asarray(values, dtype=float), dims=[AXIS], coords={AXIS: ds.coords[AXIS]})
        ds.attrs = {**self.ds.attrs, **attrs}
        return self._new(ds)

    @records_step
    def sort(self, ascending: bool = True) -> "XPSData":
        """Return a new instance with the binding energy axis sorted.

        Parameters
        ----------
        ascending : bool, optional
            Ascending when ``True`` (default); XPS is usually plotted
            descending, which is a plotting choice, not a storage one.
        """
        return self._new(self.ds.sortby(AXIS, ascending=ascending))

    @records_step
    def select_binding_energy_range(self, min_eV: Optional[float] = None, max_eV: Optional[float] = None) -> "XPSData":
        """Return a new instance restricted to a binding energy window (inclusive bounds).

        Parameters
        ----------
        min_eV, max_eV : float or None, optional
            Bounds in eV; ``None`` leaves that side open.

        Returns
        -------
        XPSData
            New instance; every stored variable is cut to the same window.

        Raises
        ------
        ValueError
            If no point falls inside the window.
        """
        energy = self.binding_energy
        mask = np.ones(energy.size, dtype=bool)
        if min_eV is not None:
            mask &= energy >= min_eV
        if max_eV is not None:
            mask &= energy <= max_eV
        if not mask.any():
            raise ValueError(
                f"No point within binding energy [{min_eV}, {max_eV}] eV: the data cover "
                f"{energy.min():.2f} to {energy.max():.2f} eV"
            )
        return self._new(self.ds.isel({AXIS: mask}))

    @records_step
    def shift_binding_energy(self, delta_eV: float) -> "XPSData":
        """Shift the binding energy axis by ``delta_eV`` (charge correction) and record the total shift.

        Parameters
        ----------
        delta_eV : float
            Added to every binding energy; the kinetic energy of the file is
            not touched since the source energy stays.

        Returns
        -------
        XPSData
            New instance; ``ds.attrs['binding_energy_shift_eV']`` holds the
            cumulative shift.
        """
        ds = self.ds.assign_coords({AXIS: self.binding_energy + float(delta_eV)})
        ds.attrs = {**self.ds.attrs,
                    "binding_energy_shift_eV": float(self.ds.attrs.get("binding_energy_shift_eV", 0.0)) + float(delta_eV)}
        return self._new(ds)

    @records_step
    def calibrate_binding_energy(self, reference_eV: float, measured_eV: float) -> "XPSData":
        """Shift the axis so a measured line lands on its reference value (e.g. adventitious C 1s at 284.8 eV).

        Parameters
        ----------
        reference_eV : float
            Where the line belongs.
        measured_eV : float
            Where it was found in this spectrum.

        Returns
        -------
        XPSData
            ``shift_binding_energy(reference_eV - measured_eV)``.
        """
        return self.shift_binding_energy(float(reference_eV) - float(measured_eV))

    @records_step
    def to_cps(self) -> "XPSData":
        """Convert counts to counts per second, ``counts / (dwell_time_s * n_scans)``.

        Returns
        -------
        XPSData
            New instance with ``data_type='cps'``; ``self`` if already cps.

        Raises
        ------
        ValueError
            If the dwell time or the number of scans is unknown, or the data
            are not counts.
        """
        if self.data_type == "cps":
            return self
        if self.data_type != "counts":
            raise ValueError(f"Only counts convert to cps, got data_type {self.data_type!r}")
        if self.dwell_time_s is None or self.n_scans is None:
            raise ValueError("to_cps needs dwell_time_s and n_scans in the attributes")
        factor = self.dwell_time_s * self.n_scans
        ds = self.ds.copy()
        for name in ("data", "background", "envelope", "peaks"):
            if name in ds:
                ds[name] = ds[name] / factor
        ds.attrs = {**self.ds.attrs, "data_type": "cps"}
        return self._new(ds)

    @records_step
    def to_counts(self) -> "XPSData":
        """The inverse of :meth:`to_cps`.

        Returns
        -------
        XPSData
            New instance with ``data_type='counts'``; ``self`` if already counts.
        """
        if self.data_type == "counts":
            return self
        if self.data_type != "cps":
            raise ValueError(f"Only cps convert back to counts, got data_type {self.data_type!r}")
        if self.dwell_time_s is None or self.n_scans is None:
            raise ValueError("to_counts needs dwell_time_s and n_scans in the attributes")
        factor = self.dwell_time_s * self.n_scans
        ds = self.ds.copy()
        for name in ("data", "background", "envelope", "peaks"):
            if name in ds:
                ds[name] = ds[name] * factor
        ds.attrs = {**self.ds.attrs, "data_type": "counts"}
        return self._new(ds)

    @records_step
    def smooth_savgol(self, window_length: int = 11, polyorder: int = 3) -> "XPSData":
        """Apply a Savitzky-Golay filter along the energy axis.

        Parameters
        ----------
        window_length : int, optional
            Window in points (default is 11).
        polyorder : int, optional
            Polynomial order (default is 3).
        """
        from scipy.signal import savgol_filter

        return self._with_values(savgol_filter(self.values, window_length, polyorder))

    @records_step
    def normalise_max(self) -> "XPSData":
        """Divide the spectrum by its maximum (``data_type`` becomes ``'arbitrary'``)."""
        maximum = float(np.nanmax(self.values))
        if maximum == 0:
            logger.warning("Maximum is zero; returning the data unchanged")
            return self
        return self._with_values(self.values / maximum, data_type="arbitrary")

    @records_step
    def subtract_background(
        self,
        method: Literal["shirley", "linear"] = "shirley",
        min_eV: Optional[float] = None,
        max_eV: Optional[float] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> "XPSData":
        """Subtract a Shirley or linear background and keep the subtracted curve as ``background``.

        The end points of the window (or of the whole spectrum) define the
        background: a straight line between them, or the Shirley curve,
        whose value at each energy is proportional to the integrated signal
        above the background towards the low kinetic energy (high binding
        energy) side. Outside the window the spectrum is left as it is.

        Parameters
        ----------
        method : {"shirley", "linear"}, optional
            The background shape (default is ``"shirley"``).
        min_eV, max_eV : float or None, optional
            The window in binding energy; ``None`` uses the spectrum's ends.
        max_iterations : int, optional
            Iterations of the Shirley recursion (default is 100).
        tolerance : float, optional
            Stop when the background changes by less than this fraction of
            its range (default is 1e-6).

        Returns
        -------
        XPSData
            New instance with the background removed from ``data`` and stored
            in ``background``; ``ds.attrs['background_method']`` names it.

        Raises
        ------
        ValueError
            For an unknown method or a window with fewer than two points.
        """
        if method not in ("shirley", "linear"):
            raise ValueError(f"method must be 'shirley' or 'linear', got {method!r}")
        energy = self.binding_energy
        values = self.values
        order = np.argsort(energy)
        energy_sorted, values_sorted = energy[order], values[order]
        mask = np.ones(energy_sorted.size, dtype=bool)
        if min_eV is not None:
            mask &= energy_sorted >= min_eV
        if max_eV is not None:
            mask &= energy_sorted <= max_eV
        if mask.sum() < 2:
            raise ValueError(f"Fewer than two points in [{min_eV}, {max_eV}] eV")
        window_energy, window_values = energy_sorted[mask], values_sorted[mask]
        if method == "linear":
            window_background = np.interp(window_energy, window_energy[[0, -1]], window_values[[0, -1]])
        else:
            window_background = _shirley(window_energy, window_values, max_iterations, tolerance)
        background_sorted = np.zeros_like(values_sorted)
        background_sorted[mask] = window_background
        background = np.empty_like(background_sorted)
        background[order] = background_sorted
        ds = self.ds.copy()
        ds["data"] = xr.DataArray(values - background, dims=[AXIS], coords={AXIS: ds.coords[AXIS]})
        ds["background"] = xr.DataArray(background, dims=[AXIS], coords={AXIS: ds.coords[AXIS]})
        ds.attrs = {**self.ds.attrs, "background_method": method,
                    "background_window_eV": [min_eV, max_eV]}
        return self._new(ds)

    @records_step
    def fit_peaks(
        self,
        peaks: Sequence[Any],
        *,
        background: Optional[Literal["shirley", "linear"]] = "shirley",
        min_eV: Optional[float] = None,
        max_eV: Optional[float] = None,
        weights: Any = None,
        **kwargs: Any,
    ) -> "XPSData":
        """Fit CasaXPS-style peaks to the region and store the fit the way a CasaXPS export is stored.

        The background is subtracted inside the window (:meth:`subtract_background`),
        the peaks are fitted to the net signal of the window
        (:func:`phd_parser.spectroscopy.xps.fitting.fit_peaks`), and the result
        is stored as CasaXPS exports it: ``data`` stays as it is, the fitted
        ``background`` is kept, and ``peaks`` and ``envelope`` are drawn on
        top of the background over the whole axis, so ``data``, ``background``,
        ``envelope`` and ``peaks`` plot together and ``peaks - background``
        are the net peaks. ``peak_area`` and the other coordinates are the
        net quantities. With ``background=None`` the data are fitted as they
        are (use it after your own :meth:`subtract_background`).

        Parameters
        ----------
        peaks : sequence of PeakSpec
            What to fit, see :class:`~phd_parser.spectroscopy.xps.fitting.PeakSpec`
            and :func:`~phd_parser.spectroscopy.xps.fitting.doublet`.
        background : {"shirley", "linear"} or None, optional
            Background subtracted inside the window before the fit (default
            is ``"shirley"``); ``None`` subtracts nothing.
        min_eV, max_eV : float or None, optional
            The window in binding energy that the background and the fit use;
            ``None`` leaves that side open.
        weights : None, "poisson" or numpy.ndarray, optional
            Residual weights. ``"poisson"`` is ``1 / sqrt(max(data, 1))`` from
            the data *before* background subtraction, right for counts; an
            array must have one weight per point of the whole axis.
        **kwargs
            Passed on to :func:`~phd_parser.spectroscopy.xps.fitting.fit_peaks`
            (``max_nfev``).

        Returns
        -------
        XPSData
            New instance with ``envelope``, the 2-D ``peaks`` with the
            coordinates ``peak_name``, ``peak_position_eV``, ``peak_fwhm_eV``,
            ``peak_area``, ``peak_area_fraction`` and ``peak_lineshape``,
            ``analysis_source = 'phd_parser'`` and ``ds.attrs['fit']``, the
            JSON-safe summary (statistics, per-peak numbers with their
            standard errors, the specifications, the background used).

        Raises
        ------
        ValueError
            If the window holds fewer points than there are free parameters,
            or for an invalid specification (see ``fitting.fit_peaks``).
        """
        from phd_parser.spectroscopy.xps.fitting import PeakSpec, fit_peaks as fit_peaks_to_arrays

        specs = [spec if isinstance(spec, PeakSpec) else PeakSpec(**dict(spec)) for spec in peaks]
        source = self if background is None else self.subtract_background(background, min_eV, max_eV)
        energy = source.binding_energy
        mask = np.ones(energy.size, dtype=bool)
        if min_eV is not None:
            mask &= energy >= min_eV
        if max_eV is not None:
            mask &= energy <= max_eV
        if isinstance(weights, str) and weights == "poisson":
            weight_array: Any = (1.0 / np.sqrt(np.maximum(self.values, 1.0)))[mask]
        elif weights is None:
            weight_array = None
        else:
            weight_array = np.asarray(weights, dtype=float)
            if weight_array.shape != energy.shape:
                raise ValueError(f"weights must have one value per point of the axis ({energy.size}), got {weight_array.shape}")
            weight_array = weight_array[mask]
        result = fit_peaks_to_arrays(energy[mask], source.values[mask], specs, weights=weight_array, **kwargs)
        if mask.sum() <= result.n_free_parameters:
            raise ValueError(f"Only {int(mask.sum())} points in the window for {result.n_free_parameters} free parameters")

        offset = source.background if background is not None else np.zeros_like(energy)
        curves = np.stack([
            spec.resolved_lineshape.evaluate(energy, peak.position_eV, peak.fwhm_eV, peak.area)
            for spec, peak in zip(result.specs, result.peaks)
        ])
        ds = source.ds.drop_vars([name for name in ("peaks", "envelope", "peak", "peak_name", "peak_position_eV",
                                                    "peak_fwhm_eV", "peak_area", "peak_area_fraction",
                                                    "peak_lineshape") if name in source.ds.variables])
        ds["data"] = xr.DataArray(self.values, dims=[AXIS], coords={AXIS: ds.coords[AXIS]})  # the curves sit on the background
        ds["envelope"] = xr.DataArray(curves.sum(axis=0) + offset, dims=[AXIS], coords={AXIS: ds.coords[AXIS]})
        ds["peaks"] = xr.DataArray(
            curves + offset[np.newaxis, :], dims=["peak", AXIS],
            coords={
                "peak": np.arange(len(result.peaks)),
                AXIS: ds.coords[AXIS],
                "peak_name": ("peak", np.asarray([peak.name for peak in result.peaks], dtype=str)),
                "peak_position_eV": ("peak", np.asarray([peak.position_eV for peak in result.peaks], dtype=float)),
                "peak_fwhm_eV": ("peak", np.asarray([peak.fwhm_eV for peak in result.peaks], dtype=float)),
                "peak_area": ("peak", np.asarray([peak.area for peak in result.peaks], dtype=float)),
                "peak_area_fraction": ("peak", np.asarray([peak.area_fraction for peak in result.peaks], dtype=float)),
                "peak_lineshape": ("peak", np.asarray([peak.lineshape for peak in result.peaks], dtype=str)),
            },
        )
        summary = result.to_dict()
        summary["background"] = {
            "method": background,
            "window_eV": [min_eV, max_eV],
            "included_in_curves": background is not None,
            "weights": weights if weights is None or isinstance(weights, str) else "array",
        }
        ds.attrs = {**source.ds.attrs, "analysis_source": "phd_parser", "fit": to_json_safe(summary)}
        if not result.success:
            logger.warning(f"Peak fit of {self.region!r} did not converge: {result.message}")
        return self._new(ds)

    # ----------------------------------------------------------------
    # Export
    # ----------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """The spectrum and every stored 1-D variable as a table indexed by binding energy in eV."""
        columns = {"data": self.values}
        if self.source_energy_eV is not None:
            columns["kinetic_energy_eV"] = self.kinetic_energy
        for name in ("transmission", "background", "envelope"):
            if name in self.ds:
                columns[name] = self.ds[name].values
        return pd.DataFrame(columns, index=pd.Index(self.binding_energy, name="binding_energy_eV"))

    def to_netcdf(self, filepath: Union[str, Path]) -> None:
        """Save to NetCDF4 (through h5netcdf), history and metadata included; :meth:`from_netcdf` reloads it.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Destination; an existing file is overwritten with a warning.
        """
        filepath = Path(filepath)
        if filepath.exists():
            logger.warning(f"Overwriting existing file: {filepath}")
        ds = self.ds.copy()
        attrs = dict(self.ds.attrs)
        if self.metadata:
            attrs["metadata"] = to_json_safe(self.metadata)
        ds.attrs = encode_attrs(attrs)
        for name in ds.variables:
            ds[name].attrs = encode_attrs(dict(ds[name].attrs))
        ds.to_netcdf(filepath, engine="h5netcdf")

    # ----------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------

    @classmethod
    def _build(
        cls,
        binding_energy_eV: npt.NDArray,
        values: npt.NDArray,
        attrs: dict[str, Any],
        extra: Optional[dict[str, npt.NDArray]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "XPSData":
        energy = np.asarray(binding_energy_eV, dtype=float)
        values = np.asarray(values, dtype=float)
        if energy.ndim != 1 or values.shape != energy.shape:
            raise ValueError(f"values {values.shape} must be 1-D and match the axis {energy.shape}")
        order = np.argsort(energy)
        coords = {AXIS: energy[order]}
        variables: dict[str, Any] = {"data": xr.DataArray(values[order], dims=[AXIS], coords=coords)}
        for name, array in (extra or {}).items():
            array = np.asarray(array, dtype=float)
            if array.shape != energy.shape:
                raise ValueError(f"{name} {array.shape} must match the axis {energy.shape}")
            variables[name] = xr.DataArray(array[order], dims=[AXIS], coords=coords)
        attrs = {"technique": "xps", **{key: value for key, value in attrs.items() if value is not None}}
        attrs.setdefault("binding_energy_shift_eV", 0.0)
        return cls(ds=xr.Dataset(variables, attrs=attrs), metadata=metadata or {})

    @classmethod
    @records_step
    def from_arrays(
        cls,
        binding_energy_eV: npt.NDArray,
        values: npt.NDArray,
        *,
        source_energy_eV: Optional[float] = None,
        data_type: DataType = "counts",
        region: Optional[str] = None,
        sample: Optional[str] = None,
        species_label: Optional[str] = None,
        dwell_time_s: Optional[float] = None,
        n_scans: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "XPSData":
        """Build a region from raw arrays.

        Parameters
        ----------
        binding_energy_eV : numpy.ndarray
            1-D binding energy axis in eV (any order; stored ascending).
        values : numpy.ndarray
            The spectrum, same length.
        source_energy_eV : float or None, optional
            Photon energy of the source in eV; needed for the kinetic energy.
        data_type : {"counts", "cps", "arbitrary"}, optional
            What the values are (default is ``"counts"``).
        region, sample, species_label : str or None, optional
            Names as the instrument would give them.
        dwell_time_s, n_scans : float, int or None, optional
            Acquisition parameters needed by :meth:`to_cps`.
        metadata : dict or None, optional
            Kept as :attr:`metadata`.

        Returns
        -------
        XPSData
            New instance.
        """
        return cls._build(
            binding_energy_eV, values,
            {"source_energy_eV": source_energy_eV, "data_type": data_type, "region": region, "sample": sample,
             "species_label": species_label, "dwell_time_s": dwell_time_s, "n_scans": n_scans},
            metadata=metadata,
        )

    @classmethod
    @records_step
    def from_netcdf(cls, filepath: Union[str, Path]) -> "XPSData":
        """Reload a region saved with :meth:`to_netcdf`.

        Parameters
        ----------
        filepath : str or pathlib.Path
            The ``.nc`` file.

        Returns
        -------
        XPSData
            The instance, its history extended by this load.
        """
        with xr.open_dataset(filepath) as handle:
            ds = handle.load()
        ds.attrs = decode_attrs(dict(ds.attrs))
        for name in ds.variables:
            ds[name].attrs = decode_attrs(dict(ds[name].attrs))
        attrs = dict(ds.attrs)
        metadata = attrs.pop("metadata", {})
        ds.attrs = attrs
        return cls(ds=ds, metadata=metadata if isinstance(metadata, dict) else {"metadata": metadata})

    @classmethod
    def from_vamas(cls, filepath: Union[str, Path], region: Optional[str] = None) -> Union[dict[str, "XPSData"], "XPSData"]:
        """Read a VAMAS (``.vms``) file: every region, or one.

        The kinetic energy axis of the file becomes binding energy with the
        block's source energy; ``Counts`` become ``data``, a ``Transmission``
        variable is kept alongside. The acquisition parameters of the block
        go into the attributes, the rest into ``metadata``.

        Parameters
        ----------
        filepath : str or pathlib.Path
            The ``.vms`` file.
        region : str or None, optional
            A block identifier (``"C1s Scan"``) to return just that region.

        Returns
        -------
        dict of str to XPSData, or XPSData
            Every region keyed by its block identifier, or the one asked for.

        Raises
        ------
        KeyError
            If ``region`` is not in the file.
        """
        from phd_parser.spectroscopy.xps.vamas import read_vamas

        raw = read_vamas(filepath)
        regions = {name: cls._from_vamas_block(name, raw["data"][name], raw["meta"]["blocks"][name])
                   for name in raw["data"]}
        if region is None:
            return regions
        if region not in regions:
            raise KeyError(f"Region {region!r} not in {Path(filepath).name}; it has {list(regions)}")
        return regions[region]

    @classmethod
    @records_step
    def _from_vamas_block(cls, name: str, block: dict[str, Any], meta: dict[str, Any]) -> "XPSData":
        """One VAMAS block as a region (recorded as the reading step of that region)."""
        source_energy = meta.get("source_energy_eV")
        if "binding_energy" in block:
            binding_energy = np.asarray(block["binding_energy"], dtype=float)
        elif source_energy is not None:
            binding_energy = _kinetic_to_binding(block["kinetic_energy"], source_energy)
        else:
            raise ValueError(f"Block {name!r} has a kinetic energy axis but no source energy to convert it")
        counts_key = next((key for key in block if key == "counts"), None)
        if counts_key is None:
            counts_key = block["variables"][0].strip().lower().replace(" ", "_")
        extra = {key: block[key] for key in ("transmission",) if key in block}
        when = meta.get("datetime")
        attrs = {
            "region": name,
            "sample": meta.get("sample"),
            "species_label": meta.get("species_label"),
            "source_energy_eV": source_energy,
            "source_label": meta.get("source_label"),
            "data_type": "counts" if counts_key == "counts" else "arbitrary",
            "dwell_time_s": meta.get("dwell_time_s"),
            "n_scans": meta.get("n_scans"),
            "pass_energy_eV": meta.get("pass_energy_eV"),
            "analyser_mode": meta.get("analyser_mode"),
            "analyser_work_function_eV": meta.get("analyser_work_function_eV"),
            "acquisition_time": when.isoformat() if isinstance(when, datetime) else None,
        }
        metadata = {key: (value.isoformat() if isinstance(value, datetime) else value) for key, value in meta.items()}
        return cls._build(binding_energy, block[counts_key], attrs, extra=extra, metadata=metadata)

    @classmethod
    @records_step
    def from_avantage_avg(cls, filepath: Union[str, Path]) -> "XPSData":
        """Read a Thermo Avantage ``.avg`` file (one region).

        The energy axis of the file is kinetic energy and becomes binding
        energy with ``DS_SOPROPID_ENERGY``. When the file says the
        transmission function was removed, the values are not raw counts and
        ``data_type`` is ``'arbitrary'``.

        Parameters
        ----------
        filepath : str or pathlib.Path
            The ``.avg`` file.

        Returns
        -------
        XPSData
            The region.
        """
        from phd_parser.spectroscopy.xps.avantage import read_avg

        raw = read_avg(filepath)
        meta = raw["meta"]
        source_energy = meta.get("source_energy_eV")
        if source_energy is None:
            raise ValueError(f"{Path(filepath).name} carries no source energy (DS_SOPROPID_ENERGY)")
        binding_energy = _kinetic_to_binding(raw["data"]["kinetic_energy"], source_energy)
        label = str(meta.get("value_label", "")).lower()
        data_type = "counts" if label == "counts" and not meta.get("transmission_removed") else "arbitrary"
        start = meta.get("start_time")
        attrs = {
            "region": meta.get("title"),
            "sample": meta.get("sample"),
            "source_energy_eV": source_energy,
            "data_type": data_type,
            "dwell_time_s": meta.get("dwell_time_s"),
            "n_scans": meta.get("n_scans"),
            "pass_energy_eV": meta.get("pass_energy_eV"),
            "transmission_removed": bool(meta.get("transmission_removed", False)),
            "acquisition_time": start.isoformat() if isinstance(start, datetime) else None,
        }
        metadata = {key: (value.isoformat() if isinstance(value, datetime) else value) for key, value in meta.items()}
        return cls._build(binding_energy, raw["data"]["values"], attrs, metadata=metadata)

    @classmethod
    @records_step
    def from_casaxps_export(cls, filepath: Union[str, Path]) -> "XPSData":
        """Read a CasaXPS tab-separated export (one region with its fit).

        ``Counts`` become ``data`` (``cps`` columns are the same divided by
        the acquisition time); the fitted peaks, the background and the
        envelope are kept as ``peaks``, ``background`` and ``envelope`` with
        ``analysis_source = 'CasaXPS'``.

        Parameters
        ----------
        filepath : str or pathlib.Path
            The export file (``.txt`` or ``.csv``).

        Returns
        -------
        XPSData
            The region.
        """
        from phd_parser.spectroscopy.xps.casaxps import read_export

        raw = read_export(filepath)
        meta = raw["meta"]
        kinetic = raw["kinetic energy"]
        binding = raw["binding energy"]
        source_energy = _first(meta.get("Characteristic Energy eV"))
        if binding is not None:
            binding_energy = np.asarray(binding.iloc[:, 0], dtype=float)
        elif kinetic is not None and source_energy is not None:
            binding_energy = _kinetic_to_binding(np.asarray(kinetic.iloc[:, 0], dtype=float), float(source_energy))
        else:
            raise ValueError(f"{Path(filepath).name}: no binding energy column and no source energy to derive one")
        frame = kinetic if kinetic is not None else binding
        columns = list(frame.columns)
        values = np.asarray(frame["Counts"] if "Counts" in columns else frame.iloc[:, 1], dtype=float)
        data_type = "counts" if "Counts" in columns else "cps"

        extra: dict[str, npt.NDArray] = {}
        peak_columns = [column for column in columns[2:] if column not in ("Background", "Envelope", "Background CPS", "Envelope CPS")]
        for name, column in (("background", "Background"), ("envelope", "Envelope")):
            if column in columns:
                extra[name] = np.asarray(frame[column], dtype=float)
            elif column + " CPS" in columns:
                extra[name] = np.asarray(frame[column + " CPS"], dtype=float)

        info = str(meta.get("info_0", ""))
        parts = [part.strip() for part in info.split(":")]
        region = parts[-1] if parts else None
        sample = parts[1] if len(parts) >= 3 else None
        acquisition = _first(meta.get("Acquisition Time s"))
        attrs = {
            "region": region,
            "sample": sample,
            "species_label": _first(meta.get("Name")) if not peak_columns or len(peak_columns) == 1 else None,
            "source_energy_eV": float(source_energy) if source_energy is not None else None,
            "data_type": data_type,
            "acquisition_time_per_point_s": float(acquisition) if acquisition is not None else None,
            "analysis_source": "CasaXPS" if extra or peak_columns else None,
        }
        result = cls._build(binding_energy, values, attrs, extra=extra, metadata={str(k): v for k, v in meta.items()})
        if peak_columns:
            result = result._with_peaks(frame, peak_columns, meta, binding_energy)
        return result

    def _with_peaks(self, frame: pd.DataFrame, peak_columns: list[str], meta: dict[str, Any],
                    binding_energy: npt.NDArray) -> "XPSData":
        order = np.argsort(np.asarray(binding_energy, dtype=float))
        curves = np.stack([np.asarray(frame[column], dtype=float)[order] for column in peak_columns])
        names = _as_list(meta.get("Name"), len(peak_columns), peak_columns)
        ds = self.ds.copy()
        ds["peaks"] = xr.DataArray(
            curves, dims=["peak", AXIS],
            coords={
                "peak": np.arange(len(peak_columns)),
                AXIS: ds.coords[AXIS],
                "peak_name": ("peak", np.asarray(names, dtype=str)),
                "peak_position_eV": ("peak", np.asarray(_as_list(meta.get("Position"), len(peak_columns)), dtype=float)),
                "peak_fwhm_eV": ("peak", np.asarray(_as_list(meta.get("FWHM"), len(peak_columns)), dtype=float)),
                "peak_area": ("peak", np.asarray(_as_list(meta.get("Area"), len(peak_columns)), dtype=float)),
                "peak_lineshape": ("peak", np.asarray(_as_list(meta.get("Lineshape"), len(peak_columns), [""]), dtype=str)),
            },
        )
        return self._new(ds)

    def __repr__(self) -> str:
        """One line: region, sample, data type, points, binding energy range."""
        energy = self.binding_energy
        span = f"{energy.min():.1f}-{energy.max():.1f} eV" if energy.size else "empty"
        extras = "".join(
            f", {name}" for name, present in (("fit", self.has_fit), ("background", self.has_background)) if present
        )
        return (f"XPSData(region={self.region!r}, sample={self.sample!r}, data_type={self.data_type}, "
                f"n_points={self.n_points}, binding_energy={span}{extras})")


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple, np.ndarray)):
        return value[0] if len(value) else None
    return value


def _as_list(value: Any, n: int, fallback: Optional[list[Any]] = None) -> list[Any]:
    if isinstance(value, (list, tuple, np.ndarray)):
        items = list(value)
    elif value is None:
        items = []
    else:
        items = [value]
    if len(items) >= n:
        return items[:n]
    if fallback is not None and len(fallback) >= n:
        return list(fallback[:n])
    return items + [np.nan] * (n - len(items))


def _shirley(energy: npt.NDArray, values: npt.NDArray, max_iterations: int, tolerance: float) -> npt.NDArray:
    """Shirley background on an ascending binding energy window: iterate ``B(E) = I_l + (I_h - I_l) A(E) / A_total``."""
    low, high = float(values[0]), float(values[-1])
    background = np.full_like(values, low, dtype=float)
    span = float(np.ptp(values)) or 1.0
    for _ in range(max_iterations):
        signal = values - background
        # cumulative area from the low binding energy end to each point
        increments = 0.5 * (signal[1:] + signal[:-1]) * np.diff(energy)
        cumulative = np.concatenate([[0.0], np.cumsum(increments)])
        total = cumulative[-1]
        if total == 0:
            break
        new_background = low + (high - low) * cumulative / total
        change = float(np.max(np.abs(new_background - background))) / span
        background = new_background
        if change < tolerance:
            break
    return background
