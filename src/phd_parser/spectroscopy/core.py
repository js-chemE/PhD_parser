"""Vibrational spectra over one wavenumber axis: the common core of infrared and Raman.

An IR band and a Raman band of the same vibration sit at the same wavenumber;
the two techniques differ in selection rules and in how the signal is
recorded, not in what the axis means. So both are one data model,
:class:`SpectroscopyData`: an immutable ``xarray.Dataset`` whose ``"data"``
variable runs over ``wavenumber`` in m⁻¹ (the Raman shift, for Raman), as a
single spectrum ``(wavenumber,)``, a series ``(scan, wavenumber)`` with
``tos`` along ``scan``, or a map ``(y, x, wavenumber)``. Every processing step
lives here: selection, time origin, background and ``data_type`` conversions,
smoothing, baseline correction, averaging, normalisation, signal to noise,
merging, Gram-Schmidt, NetCDF, SpectroChemPy.

There is one class only. ``ds.attrs["technique"]`` records what recorded
the data (``'infrared'``, ``'raman'``), the readers set it, and it decides
nothing but the default ``data_type`` and the label of a CSV column. The
Raman names (``shift``, ``select_shift_range``, the excitation wavelength
and the scattered-light axes) and the readers of both techniques are
methods of this class. ``IRData`` and ``RamanData`` are kept as aliases of
it for existing code.

Every public step is recorded in ``ds.attrs["history"]`` (see
:mod:`phd_parser.history`), which ``to_netcdf`` keeps.
"""

import logging
from functools import cached_property
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, field_validator

from phd_parser._time import TimeOriginMixin
from phd_parser.history import HistoryMixin, records_step
from phd_parser.spectroscopy._axis import _AxisMixin
from phd_parser.spectroscopy._background import _BackgroundMixin
from phd_parser.spectroscopy._baseline import _BaselineMixin
from phd_parser.spectroscopy._common import (  # noqa: F401  re-exported for the readers and old imports
    _DATA_TYPE_TO_SCP_TITLE,
    _DEFAULT_DATA_TYPE,
    _MAP_DIMS,
    _OMNIC_VLABEL_TO_DATA_TYPE,
    BACKGROUND_REFERENCED_TYPES,
    DataType,
    _dataset_from_dataarray,
)
from phd_parser.spectroscopy._io import _IOMixin
from phd_parser.spectroscopy._merge import _MergeMixin
from phd_parser.spectroscopy._selection import _SelectionMixin
from phd_parser.spectroscopy._signal import _SignalMixin
from phd_parser.spectroscopy.units import quantity_of, to_wavenumber

logger = logging.getLogger(__name__)


class SpectroscopyData(
    _AxisMixin,
    _SelectionMixin,
    TimeOriginMixin,
    _BackgroundMixin,
    _BaselineMixin,
    _SignalMixin,
    _MergeMixin,
    _IOMixin,
    HistoryMixin,
    BaseModel,
):
    """Immutable wrapper around an xarray Dataset of vibrational spectra over ``wavenumber``.

    Spectra are stored in SI units (wavenumber in m⁻¹; for Raman, the Raman
    shift). All processing methods return a new instance and are recorded in
    :attr:`history`.

    Attributes
    ----------
    ds : xr.Dataset
        Dataset containing a ``"data"`` variable with dims ``('wavenumber',)``
        for a single spectrum, ``('scan', 'wavenumber')`` for a time series
        or ``('y', 'x', 'wavenumber')`` for a map. Wavenumber coordinate is
        in m⁻¹. An optional ``'tos'`` coordinate (elapsed seconds, attached
        to the ``scan`` dim), a ``'background'`` variable (1-D,
        single_beam), and a ``'baseline'`` variable (same dims as ``'data'``)
        may also be present.  Dataset-level attrs hold ``technique``
        (``'infrared'``, ``'raman'`` or absent), ``data_type``,
        ``wavenumber_unit``, ``tos_start`` (ISO string),
        ``excitation_wavelength_nm`` (Raman), ``history`` and provenance keys
        written by processing methods.
    metadata : dict
        Raw metadata of the source file, as the reader found it. Saved by
        :meth:`to_netcdf` as JSON.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        ignored_types=(cached_property,),
    )

    ds: xr.Dataset = Field(
        description=(
            "xarray Dataset with a 'data' variable with dims ('wavenumber',), ('scan', 'wavenumber') "
            "or ('y', 'x', 'wavenumber'). Wavenumber in m⁻¹. Optional coord: 'tos' (seconds). "
            "Attrs at dataset level."
        )
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw metadata extracted from the source file.",
    )

    @field_validator("ds", mode="before")
    @classmethod
    def validate_ds(cls, v: Any) -> xr.Dataset:
        """Check the layout of ``ds``: a ``data`` variable over ``wavenumber``, optional background and baseline."""
        if not isinstance(v, xr.Dataset):
            raise TypeError(f"'ds' must be an xr.Dataset, got {type(v)}")
        if "data" not in v:
            raise ValueError("Dataset must contain a 'data' variable")
        data_var = v["data"]
        if "wavenumber" not in data_var.dims:
            raise ValueError("Dataset 'data' variable must have a 'wavenumber' dimension")
        if tuple(data_var.dims) != ("y", "x", "wavenumber"):
            if data_var.ndim not in (1, 2):
                raise ValueError(f"Dataset 'data' variable must be 1-D, 2-D or a (y, x, wavenumber) map, got dims {data_var.dims}")
            if data_var.ndim == 2 and data_var.dims[0] != "scan":
                raise ValueError("2-D Dataset 'data' variable must have dims ('scan', 'wavenumber')")
        if "background" in v:
            cls._validate_background_var(v)
        if "baseline" in v:
            cls._validate_baseline_var(v)
        excitation = v.attrs.get("excitation_wavelength_nm")
        if excitation is not None and float(excitation) <= 0:
            raise ValueError(f"excitation_wavelength_nm must be positive, got {excitation}")
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

    def _new(self, ds: xr.Dataset) -> "SpectroscopyData":
        """A new instance around ``ds``, keeping the other fields (the metadata)."""
        return SpectroscopyData(ds=ds, metadata=self.metadata)

    @classmethod
    @records_step
    def from_arrays(
        cls,
        axis: npt.NDArray,
        values: npt.NDArray,
        *,
        axis_unit: str = "cm^-1",
        tos: Optional[npt.NDArray] = None,
        tos_start: Optional[Union[pd.Timestamp, str]] = None,
        data_type: Optional[DataType] = None,
        technique: Optional[str] = None,
        excitation_wavelength_nm: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "SpectroscopyData":
        """Construct a SpectroscopyData from raw NumPy arrays, the axis in any spectral unit.

        Everything after ``values`` is keyword-only. The axis is stored as a
        wavenumber in m⁻¹ whatever unit it came in; :meth:`axis` reads it
        back in any unit.

        Parameters
        ----------
        axis : numpy.ndarray
            1-D spectral axis in ``axis_unit``: a wavenumber (for Raman, the
            Raman shift), a wavelength, a frequency or a photon energy. A
            wavelength given together with ``excitation_wavelength_nm`` is
            read as the scattered light of a Raman spectrum and stored as the
            shift.
        axis_unit : str, optional
            Unit of ``axis`` (default is ``'cm^-1'``), any of
            :data:`phd_parser.spectroscopy.units.SPECTRAL_UNITS`.
        values : numpy.ndarray
            Spectral values.  Shape ``(n_wavenumber,)`` for a single spectrum
            or ``(n_scan, n_wavenumber)`` for a time series.
        tos : numpy.ndarray or None, optional
            1-D array of elapsed seconds per scan of a series; ignored for a
            single spectrum (default is ``None``).
        tos_start : pandas.Timestamp or str or None, optional
            Absolute start time of the measurement (default is ``None``).
        data_type : str or None, optional
            What the values are (``'single_beam'``, ``'absorbance'``, ...,
            ``'intensity'``, ``'counts'``).  Default: the technique's,
            ``'single_beam'`` for infrared and ``'intensity'`` for Raman, else
            ``None``.
        technique : str or None, optional
            What recorded the data, ``'infrared'`` or ``'raman'``.  Default:
            ``'raman'`` when ``excitation_wavelength_nm`` is given, else none.
        excitation_wavelength_nm : float or None, optional
            Excitation laser wavelength in nm (Raman); gives the
            scattered-light axes.
        metadata : dict or None, optional
            Raw metadata of the source, kept in :attr:`metadata`.

        Returns
        -------
        SpectroscopyData
            New instance built from the provided arrays.

        Raises
        ------
        ValueError
            If ``wavenumber_per_cm`` is not 1-D, if ``values`` shape is
            inconsistent with the wavenumber axis, if ``tos`` size does not
            match the number of scans, or if the excitation wavelength is not
            positive.
        """
        if excitation_wavelength_nm is not None and float(excitation_wavelength_nm) <= 0:
            raise ValueError(f"excitation_wavelength_nm must be positive, got {excitation_wavelength_nm}")
        zero_si = None
        if quantity_of(axis_unit) == "wavelength" and excitation_wavelength_nm is not None:
            zero_si = 1e9 / float(excitation_wavelength_nm)
        wavenumber_si = np.asarray(to_wavenumber(np.asarray(axis, dtype=float), axis_unit, zero_si), dtype=float)
        values = np.asarray(values, dtype=float)

        if wavenumber_si.ndim != 1:
            raise ValueError("axis must be 1-D")

        if values.ndim == 1:
            if values.size != wavenumber_si.size:
                raise ValueError(f"values size ({values.size}) != wavenumber size ({wavenumber_si.size})")
            tos = None
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

        if technique is None and excitation_wavelength_nm is not None:
            technique = "raman"

        attrs: dict[str, Any] = {"data_type": data_type}
        if technique is not None:
            attrs["technique"] = technique
        if excitation_wavelength_nm is not None:
            attrs["excitation_wavelength_nm"] = float(excitation_wavelength_nm)
        if tos_start is not None:
            attrs["tos_start"] = pd.Timestamp(tos_start).isoformat()

        ds = cls._build_ds(wavenumber_si, values, tos=tos, attrs=attrs)
        return cls(ds=ds, metadata=metadata or {})

    def __len__(self) -> int:
        """Number of spectra: scans of a series, pixels of a map, 1 for a single spectrum."""
        if self.is_map:
            return int(self.ds.sizes["y"] * self.ds.sizes["x"])
        return self.ds.sizes.get("scan", 1)

    @records_step
    def __add__(self, other: "SpectroscopyData") -> "SpectroscopyData":
        """Add two spectra point by point (same axis and shape); the left operand's attributes win."""
        if not isinstance(other, SpectroscopyData):
            return NotImplemented
        self._check_compatible(other, "add")
        self._warn_drop_baseline("combining two SpectroscopyData via __add__ invalidates either baseline")
        other._warn_drop_baseline("combining two SpectroscopyData via __add__ invalidates either baseline")
        new_attrs = {**other.ds.attrs, **self.ds.attrs}  # the left operand's attributes win
        ds_new = self._build_ds(self.wavenumber, self.values + other.values, tos=self.tos, attrs=new_attrs)
        return self._new(ds=self._carry_auxiliary(ds_new))

    @records_step
    def __sub__(self, other: "SpectroscopyData") -> "SpectroscopyData":
        """Subtract two spectra point by point (same axis and shape); the left operand's attributes win."""
        if not isinstance(other, SpectroscopyData):
            return NotImplemented
        self._check_compatible(other, "subtract")
        self._warn_drop_baseline("combining two SpectroscopyData via __sub__ invalidates either baseline")
        other._warn_drop_baseline("combining two SpectroscopyData via __sub__ invalidates either baseline")
        new_attrs = {**other.ds.attrs, **self.ds.attrs}  # the left operand's attributes win
        ds_new = self._build_ds(self.wavenumber, self.values - other.values, tos=self.tos, attrs=new_attrs)
        return self._new(ds=self._carry_auxiliary(ds_new))

    def _check_compatible(self, other: "SpectroscopyData", op: str) -> None:
        if self.wavenumber.shape != other.wavenumber.shape or not np.allclose(self.wavenumber, other.wavenumber):
            raise ValueError(f"Cannot {op} SpectroscopyData with different wavenumber axes")
        if self.ndim != other.ndim:
            raise ValueError(f"Cannot {op} SpectroscopyData with different number of dimensions")
        if self.ndim == 2 and self.shape[0] != other.shape[0]:
            raise ValueError(f"Cannot {op} 2-D SpectroscopyData with different number of scans")

    def _map_coords_for(self, ds: xr.Dataset) -> dict[str, Any]:
        return {
            dim: self.ds.coords[dim].values
            for dim in _MAP_DIMS
            if dim in self.ds.coords and dim in ds.dims and ds.sizes[dim] == self.ds.sizes[dim]
        }

    def _carry_auxiliary(self, ds: xr.Dataset) -> xr.Dataset:
        """Copy background and map coordinates from self into ds (wavenumber axis must match)."""
        if "background" in self.ds:
            ds = ds.assign({"background": self.ds["background"]})
        return ds.assign_coords(self._map_coords_for(ds))

    def _slice_auxiliary_to(self, ds: xr.Dataset) -> xr.Dataset:
        """Copy background (sliced to ds's wavenumber coordinate) and map coordinates into ds."""
        if "background" in self.ds:
            ds = ds.assign({"background": self.ds["background"].sel(wavenumber=ds.coords["wavenumber"])})
        return ds.assign_coords(self._map_coords_for(ds))

    def __repr__(self) -> str:
        """One line: data type, technique, shape, axis range, time, excitation, background and baseline."""
        wn = self.wavenumber_per_cm
        wn_range = f"{wn.min():.1f}–{wn.max():.1f} cm^-1" if wn.size else "empty"
        tos_info = f", tos={self.tos[0]:.1f}–{self.tos[-1]:.1f}s" if self.tos is not None and self.tos.size else ""
        ts_info = f", tos_start={self.tos_start}" if self.tos_start is not None else ""
        bg_info = f", background={self.background_data_type}" if self.has_background else ""
        bl_info = ", baseline=stored" if self.has_baseline else ""
        technique_info = f", technique={self.technique}" if self.technique is not None else ""
        excitation = self.excitation_wavelength_nm
        excitation_info = f", excitation={excitation:.1f} nm" if excitation is not None else ""
        return (
            f"SpectroscopyData(data_type={self.data_type}{technique_info}, shape={self.shape}, "
            f"wavenumber={wn_range}{tos_info}{ts_info}{excitation_info}{bg_info}{bl_info})"
        )

    @classmethod
    def _build_ds(
        cls,
        wavenumber_si: npt.NDArray,
        values: npt.NDArray,
        tos: Optional[npt.NDArray] = None,
        attrs: Optional[dict[str, Any]] = None,
        sort_by_tos: bool = True,
    ) -> xr.Dataset:
        """Build the Dataset of a spectrum, a series or a map; a series is stored in time order.

        Parameters
        ----------
        wavenumber_si : numpy.ndarray
            The axis in m⁻¹.
        values : numpy.ndarray
            ``(n_wavenumber,)``, ``(n_scan, n_wavenumber)`` or ``(y, x, n_wavenumber)``.
        tos : numpy.ndarray or None, optional
            Elapsed seconds per scan of a series.
        attrs : dict or None, optional
            Dataset attributes; ``technique`` sets the default ``data_type``.
        sort_by_tos : bool, optional
            Sort the scans of a series by ``tos`` when they are not in time
            order (default is ``True``). A series out of time order is what a
            mispaired reader produces, so nothing downstream should ever see
            one; ``merge(sort=False)`` is the one caller that keeps blocks.

        Returns
        -------
        xarray.Dataset
            With the ``data`` variable and, for a series, the ``scan`` and
            ``tos`` coordinates.
        """
        if tos is not None and values.ndim == 2 and sort_by_tos:
            tos = np.asarray(tos, dtype=float)
            if tos.size == values.shape[0] and np.any(np.diff(tos) < 0):
                order = np.argsort(tos, kind="stable")
                logger.info("Scans were not in time order; sorted %d scans by tos.", tos.size)
                values, tos = values[order], tos[order]
        coords: dict[str, Any] = {"wavenumber": wavenumber_si}
        dims: list[str]

        attrs = {} if attrs is None else dict(attrs)  # never edit the caller's dict

        attrs["wavenumber_unit"] = "m^-1"
        if attrs.get("technique") is None:
            attrs.pop("technique", None)
        if attrs.get("data_type") is None:
            attrs["data_type"] = _DEFAULT_DATA_TYPE.get(attrs.get("technique"))

        if values.ndim == 1:
            dims = ["wavenumber"]
        elif values.ndim == 2:
            dims = ["scan", "wavenumber"]
            coords["scan"] = np.arange(values.shape[0])
            if tos is not None:
                coords["tos"] = ("scan", np.asarray(tos, dtype=float))
                attrs["tos_unit"] = "s"
            else:
                logger.warning("No 'tos' provided for 2-D data; 'tos' coordinate will be missing")
        elif values.ndim == 3:
            dims = ["y", "x", "wavenumber"]
        else:
            raise ValueError(f"values must be 1-D, 2-D or a 3-D map, got shape {values.shape}")

        data_var = xr.DataArray(data=values, coords=coords, dims=dims, name="data")
        ds = xr.Dataset({"data": data_var}, attrs=attrs)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Built Dataset with dims=%s, coords=%s, tos_start=%s, attribute_keys=%s",
                dict(ds.sizes), list(ds.coords), attrs.get("tos_start"), list(attrs),
            )
        return ds
