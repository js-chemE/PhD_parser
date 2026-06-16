import logging
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scipy import constants as const

from phd_parser.raman import btc655n, renishaw

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
VLabel = Literal["intensity", "counts", "counts_per_second", "arbitrary"]


# ===========================================================================
# RamanData
# ===========================================================================

class RamanData(BaseModel):
    """Pydantic wrapper around an xarray DataArray for Raman spectroscopy data.

    The Raman shift axis is stored in SI units (m⁻¹), Stokes-positive convention.
    Three data layouts are supported:

    - ``(shift,)`` — single spectrum
    - ``(scan, shift)`` — time-series of spectra
    - ``(y, x, shift)`` — spatial map

    For time-series data the optional ``tos`` coordinate holds elapsed seconds
    since the start of the experiment.  All processing methods return a new
    ``RamanData`` instance rather than mutating the original.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        ignored_types=(cached_property,),
    )

    # ----------------------------------------------------------------
    # Fields
    # ----------------------------------------------------------------

    da: xr.DataArray = Field(
        description=(
            "xarray DataArray. Must have a 'shift' dimension (Raman shift in m⁻¹, "
            "Stokes convention: positive for Stokes lines). "
            "Optional dims: 'scan' (time series), 'x'/'y' (spatial map). "
            "Optional coords: 'tos' (seconds)."
        )
    )
    excitation_wavelength_nm: float = Field(
        description="Excitation laser wavelength in nm."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw metadata extracted from the source file.",
    )
    values_label: VLabel = Field(
        default="intensity",
        description="Nature of the intensity values.",
    )

    # ----------------------------------------------------------------
    # Validators
    # ----------------------------------------------------------------

    @field_validator("da", mode="before")
    @classmethod
    def validate_da(cls, v: Any) -> xr.DataArray:
        if not isinstance(v, xr.DataArray):
            raise TypeError(f"'da' must be an xr.DataArray, got {type(v)}")
        if "shift" not in v.dims:
            raise ValueError(
                "DataArray must have a 'shift' dimension "
                "(Raman shift in m⁻¹, Stokes > 0)"
            )
        allowed_ndim = {1, 2, 3}  # (shift,), (scan, shift), (x, y, shift)
        if v.ndim not in allowed_ndim:
            raise ValueError(
                f"DataArray must be 1-, 2-, or 3-D, got {v.ndim}-D"
            )
        return v

    @field_validator("excitation_wavelength_nm", mode="before")
    @classmethod
    def validate_excitation(cls, v: Any) -> float:
        v = float(v)
        if v <= 0:
            raise ValueError(f"excitation_wavelength_nm must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def set_default_attrs(self) -> "RamanData":
        self.da.attrs.setdefault("values_label", self.values_label)
        self.da.attrs.setdefault("shift_units", "m^-1")
        self.da.attrs.setdefault(
            "excitation_wavelength_nm", self.excitation_wavelength_nm
        )
        return self

    # ----------------------------------------------------------------
    # Core properties
    # ----------------------------------------------------------------

    @property
    def ndim(self) -> int:
        """Number of dimensions of the underlying DataArray.

        Returns
        -------
        int
            1 for a single spectrum, 2 for a time-series, 3 for a map.
        """
        return self.da.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying DataArray.

        Returns
        -------
        tuple of int
            Tuple of dimension sizes in the same order as ``da.dims``.
        """
        return tuple(self.da.shape)

    @property
    def values(self) -> npt.NDArray:
        """Raw intensity array as a NumPy ndarray.

        Returns
        -------
        numpy.ndarray
            Array of intensity values with the same shape as ``da``.
        """
        return self.da.values

    @property
    def n_spectral(self) -> int:
        """Number of spectral (shift) points.

        Returns
        -------
        int
            Length of the ``shift`` dimension.
        """
        return self.da.sizes["shift"]

    @property
    def shift(self) -> npt.NDArray:
        """Raman shift coordinate in SI units (m⁻¹), Stokes positive.

        Returns
        -------
        numpy.ndarray
            1-D array of Raman shift values in m⁻¹.
        """
        return self.da.coords["shift"].values

    @property
    def shift_per_cm(self) -> npt.NDArray:
        """Raman shift coordinate in conventional cm⁻¹, Stokes positive.

        Returns
        -------
        numpy.ndarray
            1-D array of Raman shift values in cm⁻¹.
        """
        return self.shift / 100.0

    @property
    def tos(self) -> Optional[npt.NDArray]:
        """Time-of-scan coordinate in seconds, or ``None`` if absent.

        Returns
        -------
        numpy.ndarray or None
            1-D array of elapsed seconds since run start, aligned with the
            ``scan`` dimension, or ``None`` when no ``tos`` coordinate exists.
        """
        if "tos" in self.da.coords:
            return self.da.coords["tos"].values
        return None

    # ----------------------------------------------------------------
    # Cached derived spectral axes
    # ----------------------------------------------------------------

    @cached_property
    def excitation_wavenumber_per_cm(self) -> float:
        """Excitation laser wavenumber in cm⁻¹.

        Returns
        -------
        float
            Wavenumber of the excitation laser in cm⁻¹, derived from
            ``excitation_wavelength_nm``.
        """
        return 1e7 / self.excitation_wavelength_nm

    @cached_property
    def excitation_wavenumber(self) -> float:
        """Excitation laser wavenumber in m⁻¹.

        Returns
        -------
        float
            Wavenumber of the excitation laser in m⁻¹.
        """
        return self.excitation_wavenumber_per_cm * 100.0

    @cached_property
    def wavenumber(self) -> npt.NDArray:
        """Absolute scattered wavenumber in SI units (m⁻¹).

        For Stokes scattering: ``wavenumber_scattered = wavenumber_excitation - shift``.

        Returns
        -------
        numpy.ndarray
            1-D array of absolute scattered wavenumbers in m⁻¹.
        """
        return self.excitation_wavenumber - self.shift

    @cached_property
    def wavenumber_per_cm(self) -> npt.NDArray:
        """Absolute scattered wavenumber in cm⁻¹.

        Returns
        -------
        numpy.ndarray
            1-D array of absolute scattered wavenumbers in cm⁻¹.
        """
        return self.wavenumber / 100.0

    @cached_property
    def wavelength(self) -> npt.NDArray:
        """Scattered photon wavelength in metres.

        Returns
        -------
        numpy.ndarray
            1-D array of scattered wavelengths in m.
        """
        return 1.0 / self.wavenumber

    @cached_property
    def wavelength_nm(self) -> npt.NDArray:
        """Scattered photon wavelength in nanometres.

        Returns
        -------
        numpy.ndarray
            1-D array of scattered wavelengths in nm.
        """
        return self.wavelength * 1e9

    @cached_property
    def frequency(self) -> npt.NDArray:
        """Scattered photon frequency in Hz.

        Returns
        -------
        numpy.ndarray
            1-D array of scattered photon frequencies in Hz.
        """
        return self.wavenumber * const.c

    # ----------------------------------------------------------------
    # Spectral indexing
    # ----------------------------------------------------------------

    def get_scan(self, scan_index: int) -> npt.NDArray:
        """Return the intensity array for a single scan by index.

        Parameters
        ----------
        scan_index : int
            Zero-based index along the ``scan`` dimension.

        Returns
        -------
        numpy.ndarray
            1-D intensity array of length ``n_spectral``.

        Raises
        ------
        ValueError
            When the DataArray has no ``scan`` dimension.
        IndexError
            When ``scan_index`` is outside ``[0, n_scans)``.
        """
        if "scan" not in self.da.dims:
            raise ValueError("get_scan requires a 'scan' dimension")
        n = self.da.sizes["scan"]
        if not (0 <= scan_index < n):
            raise IndexError(f"scan_index {scan_index} out of bounds for {n} scans")
        return self.da.isel(scan=scan_index).values

    def get_evolution(
        self,
        shift_per_cm: Union[float, list[float], npt.NDArray],
        method: Literal["nearest", "linear"] = "nearest",
        tolerance_per_cm: Optional[float] = None,
    ) -> xr.DataArray:
        """Return intensity vs. scan at one or more fixed Raman shift positions.

        Parameters
        ----------
        shift_per_cm : float or list of float or numpy.ndarray
            Target Raman shift(s) in cm⁻¹.
        method : {"nearest", "linear"}, optional
            Interpolation method passed to ``xr.DataArray.sel``
            (default is "nearest").
        tolerance_per_cm : float, optional
            Maximum allowed distance in cm⁻¹ between a requested shift and the
            nearest grid point.  Raises ``ValueError`` when exceeded.

        Returns
        -------
        xarray.DataArray
            DataArray with a ``scan`` dimension (and optionally a ``shift``
            dimension when multiple targets are requested).

        Raises
        ------
        ValueError
            When the DataArray has no ``scan`` dimension, or when a requested
            shift exceeds ``tolerance_per_cm`` from the nearest grid point.
        """
        if "scan" not in self.da.dims:
            raise ValueError("get_evolution requires a 'scan' dimension")

        targets_si = np.atleast_1d(np.asarray(shift_per_cm, dtype=float)) * 100.0

        if tolerance_per_cm is not None:
            tol_si = tolerance_per_cm * 100.0
            for t in targets_si:
                nearest_dist = float(np.abs(self.shift - t).min())
                if nearest_dist > tol_si:
                    raise ValueError(
                        f"Requested shift {t / 100:.1f} cm⁻¹ is "
                        f"{nearest_dist / 100:.1f} cm⁻¹ from the nearest grid point "
                        f"(tolerance: {tolerance_per_cm:.1f} cm⁻¹)"
                    )

        return self.da.sel(shift=targets_si, method=method)

    def get_map_spectrum(self, x: int, y: int) -> npt.NDArray:
        """Return the spectrum at a specific map pixel.

        Parameters
        ----------
        x : int
            Zero-based index along the ``x`` dimension.
        y : int
            Zero-based index along the ``y`` dimension.

        Returns
        -------
        numpy.ndarray
            1-D intensity array of length ``n_spectral``.

        Raises
        ------
        ValueError
            When the DataArray does not have both ``x`` and ``y`` dimensions.
        """
        if "x" not in self.da.dims or "y" not in self.da.dims:
            raise ValueError("get_map_spectrum requires 'x' and 'y' dimensions")
        return self.da.isel(x=x, y=y).values

    # ----------------------------------------------------------------
    # immutable transformations
    # ----------------------------------------------------------------

    def sort(self, ascending: bool = True) -> "RamanData":
        """Return a new RamanData with the shift axis sorted.

        The original object is not modified.  Immutable style is used
        because cached_property values are tied to the instance.

        Parameters
        ----------
        ascending : bool, optional
            Sort in ascending order when ``True`` (default), descending when
            ``False``.

        Returns
        -------
        RamanData
            New instance with the ``shift`` coordinate sorted.
        """
        da_sorted = self.da.sortby("shift", ascending=ascending)
        return RamanData(
            da=da_sorted,
            excitation_wavelength_nm=self.excitation_wavelength_nm,
            metadata=self.metadata,
            values_label=self.values_label,
        )

    def select_shift_range(
        self,
        min_shift_per_cm: Optional[float] = None,
        max_shift_per_cm: Optional[float] = None,
    ) -> "RamanData":
        """Return a new RamanData restricted to the specified shift range.

        The original object is not modified.  Immutable style is used
        because cached_property values are tied to the instance.

        Parameters
        ----------
        min_shift_per_cm : float, optional
            Lower bound of the Raman shift range in cm⁻¹.  No lower clipping
            is applied when omitted.
        max_shift_per_cm : float, optional
            Upper bound of the Raman shift range in cm⁻¹.  No upper clipping
            is applied when omitted.

        Returns
        -------
        RamanData
            New instance containing only the spectral points within the
            requested range.
        """
        da = self.da
        if min_shift_per_cm is not None:
            da = da.sel(shift=slice(min_shift_per_cm * 100.0, None))
        if max_shift_per_cm is not None:
            da = da.sel(shift=slice(None, max_shift_per_cm * 100.0))


        da_selected = self._build_da(
            shift_si=da.coords["shift"].values,
            values=da.values,
            values_label=self.values_label,
            tos=self.tos,
            metadata=self.metadata,
        )
        return RamanData(
            da=da_selected,
            excitation_wavelength_nm=self.excitation_wavelength_nm,
            metadata=self.metadata,
            values_label=self.values_label,
        )

    # ----------------------------------------------------------------
    # Export
    # ----------------------------------------------------------------

    def to_csv(
        self,
        filepath: Union[str, Path],
        shift_units: Literal["cm-1", "m-1"] = "cm-1",
    ) -> None:
        """Write the spectra to a CSV file.

        Map data (3-D) cannot be exported to CSV; use :meth:`to_netcdf` instead.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Destination file path.  The file is overwritten if it already exists.
        shift_units : {"cm-1", "m-1"}, optional
            Units for the shift index column (default is "cm-1").

        Raises
        ------
        ValueError
            When the DataArray is 3-D (map data).
        """
        filepath = Path(filepath)

        if self.da.ndim > 2:
            raise ValueError(
                "CSV export is not supported for map data (ndim > 2). "
                "Use to_netcdf() instead."
            )

        shift = self.shift_per_cm if shift_units == "cm-1" else self.shift
        shift_label = f"raman_shift [{shift_units}]"

        if self.ndim == 1:
            df = pd.DataFrame(
                {self.values_label: self.values},
                index=pd.Index(shift, name=shift_label),
            )
        else:
            tos = self.tos
            col_labels = (
                [f"tos_{t:.2f}s" for t in tos]
                if tos is not None
                else [f"scan_{i}" for i in range(self.da.sizes["scan"])]
            )
            df = pd.DataFrame(
                self.values.T,
                index=pd.Index(shift, name=shift_label),
                columns=col_labels,
            )

        df.to_csv(filepath)
        logger.debug("Saved CSV → %s", filepath)

    def to_netcdf(self, filepath: Union[str, Path]) -> None:
        """Write the DataArray to a NetCDF file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Destination file path.  The file is overwritten if it already exists.
        """
        self.da.to_netcdf(filepath)
        logger.debug("Saved NetCDF → %s", filepath)

    # ----------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        shift_per_cm: npt.NDArray,
        values: npt.NDArray,
        excitation_wavelength_nm: float,
        values_label: VLabel = "intensity",
        tos: Optional[npt.NDArray] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "RamanData":
        """Construct a RamanData from raw NumPy arrays.

        Parameters
        ----------
        shift_per_cm : numpy.ndarray
            1-D array of Raman shift values in cm⁻¹.
        values : numpy.ndarray
            1-D (single spectrum) or 2-D ``(n_scans, n_spectral)`` intensity
            array.  The last axis must match ``shift_per_cm``.
        excitation_wavelength_nm : float
            Excitation laser wavelength in nm.
        values_label : {"intensity", "counts", "counts_per_second", "arbitrary"}, optional
            Label describing the physical meaning of the intensity values
            (default is "intensity").
        tos : numpy.ndarray, optional
            1-D array of elapsed seconds since run start, one value per scan.
            Ignored for 1-D ``values``.
        metadata : dict, optional
            Arbitrary provenance metadata stored on the instance.

        Returns
        -------
        RamanData
            New instance built from the provided arrays.

        Raises
        ------
        ValueError
            When ``shift_per_cm`` is not 1-D, when ``values`` is not 1-D or
            2-D, when ``values.shape[-1]`` does not match ``shift_per_cm``,
            or when the size of ``tos`` does not match ``values.shape[0]``.
        """
        shift_si = np.asarray(shift_per_cm, dtype=float) * 100.0
        values = np.asarray(values, dtype=float)

        if shift_si.ndim != 1:
            raise ValueError("shift_per_cm must be 1-D")
        if values.ndim not in (1, 2):
            raise ValueError(f"values must be 1-D or 2-D, got shape {values.shape}")
        if values.shape[-1] != shift_si.size:
            raise ValueError(
                f"values.shape[-1] ({values.shape[-1]}) != shift size ({shift_si.size})"
            )
        if values.ndim == 2 and tos is not None:
            tos = np.asarray(tos, dtype=float)
            if tos.size != values.shape[0]:
                raise ValueError(
                    f"tos size ({tos.size}) != values.shape[0] ({values.shape[0]})"
                )

        da = cls._build_da(shift_si, values, values_label, tos, metadata)
        return cls(
            da=da,
            excitation_wavelength_nm=float(excitation_wavelength_nm),
            metadata=metadata or {},
            values_label=values_label,
        )

    @classmethod
    def from_netcdf(cls, filepath: Union[str, Path]) -> "RamanData":
        """Load a RamanData from a NetCDF file written by :meth:`to_netcdf`.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to the NetCDF file.

        Returns
        -------
        RamanData
            Instance reconstructed from the NetCDF file.

        Raises
        ------
        ValueError
            When the NetCDF file does not contain the
            ``excitation_wavelength_nm`` attribute.
        """
        da = xr.open_dataarray(filepath)
        excitation_nm = da.attrs.get("excitation_wavelength_nm")
        if excitation_nm is None:
            raise ValueError(
                "NetCDF file is missing 'excitation_wavelength_nm' attribute. "
                "Provide it manually via from_arrays()."
            )
        return cls(
            da=da,
            excitation_wavelength_nm=float(excitation_nm),
            metadata=dict(da.attrs),
        )

    @classmethod
    def from_btc655n_export(
        cls,
        filepath: Union[str, Path],
        y_key: btc655n.Y_KEYS = "Raw data #1",
        remove_empty: bool = True,
    ) -> "RamanData":
        """Load a RamanData from a B&W Tek BTC655N spectrometer export file.

        The BTC655N exports wavelength in nm; this constructor converts those
        values to Raman shift in cm⁻¹ using the excitation wavelength found
        in the file metadata.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to the BTC655N text export file.
        y_key : str, optional
            Column name to use as the intensity axis (default is
            "Raw data #1").  Must be one of the ``Y_KEYS`` literals defined
            in :mod:`phd_parser.raman.btc655n`.
        remove_empty : bool, optional
            Drop rows with no wavelength data when ``True`` (default).

        Returns
        -------
        RamanData
            Instance loaded from the BTC655N export.

        Raises
        ------
        ValueError
            When the export metadata does not contain ``laser_wavelength``.
        """
        raw = btc655n.read_export(filepath, remove_empty=remove_empty)

        excitation_nm = raw["meta"].get("laser_wavelength")
        if excitation_nm is None:
            raise ValueError(
                "BTC655N export is missing 'laser_wavelength' in metadata. "
                "Cannot compute Raman shift without excitation wavelength."
            )

        # BTC655N reports wavelength in nm → convert to Raman shift in cm⁻¹
        wavelength_nm = np.asarray(raw["data"]["Wavelength"], dtype=float)
        wavenumber_per_cm = 1e7 / wavelength_nm                             # scattered (cm⁻¹)
        excitation_per_cm = 1e7 / float(excitation_nm)                      # excitation (cm⁻¹)
        shift_per_cm = excitation_per_cm - wavenumber_per_cm                # Stokes shift (cm⁻¹)

        return cls.from_arrays(
            shift_per_cm=shift_per_cm,
            values=np.asarray(raw["data"][y_key], dtype=float),
            excitation_wavelength_nm=float(excitation_nm),
            metadata=raw["meta"],
        )

    @classmethod
    def from_renishaw_txt(
        cls,
        filepath: Union[str, Path],
        excitation_wavelength_nm: float,
    ) -> "RamanData":
        """Load a RamanData from a Renishaw WiRe plain-text export.

        Renishaw txt exports contain the Raman shift axis directly in cm⁻¹,
        so the excitation wavelength must be supplied separately.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to the Renishaw ``.txt`` export file.
        excitation_wavelength_nm : float
            Excitation laser wavelength in nm.

        Returns
        -------
        RamanData
            Instance loaded from the Renishaw txt export.
        """
        raw = renishaw.read_export_txt(Path(filepath))

        # Renishaw txt exports the shift axis directly in cm⁻¹
        shift_per_cm = np.asarray(raw["data"]["wavenumber"], dtype=float)

        return cls.from_arrays(
            shift_per_cm=shift_per_cm,
            values=np.asarray(raw["data"]["intensity"], dtype=float),
            excitation_wavelength_nm=excitation_wavelength_nm,
            metadata=raw["meta"],
        )

    @classmethod
    def from_renishaw_wdf(cls, filepath: Union[str, Path]) -> "RamanData":
        """Load a RamanData from a Renishaw WiRe binary WDF file.

        The excitation wavelength and shift axis are read directly from the
        WDF header.  Both single spectra, time-series (2-D), and spatial maps
        (3-D) are supported.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to the Renishaw ``.wdf`` binary file.

        Returns
        -------
        RamanData
            Instance loaded from the WDF file.

        Raises
        ------
        ValueError
            When the WDF x-axis unit is not ``RamanShift``, when the data
            unit is not ``Counts``, or when the spectral axis length does not
            match the data array.
        """
        filepath = Path(filepath)
        wdf = renishaw.read_export_wdf(filepath)

        if wdf.x_unit != renishaw.UnitType.RamanShift:
            raise ValueError(
                f"Expected x_unit RamanShift, got {wdf.x_unit}. "
                "Cannot interpret data as Raman shift."
            )
        if wdf.data_unit != renishaw.UnitType.Counts:
            raise ValueError(
                f"Expected data_unit Counts, got {wdf.data_unit}."
            )

        # WDF stores the shift axis directly in cm⁻¹
        shift_per_cm = np.asarray(wdf.wavenumber, dtype=float)
        shift_si = shift_per_cm * 100.0

        # Excitation wavelength from WDF header (laser_cm1 is in cm⁻¹)
        excitation_nm = 1e7 / float(wdf.laser_cm1)

        data = np.asarray(wdf.data, dtype=float)

        # ---- handle map data (3-D) ----------------------------------------
        if data.ndim == 3:
            # Shape from WDF is typically (n_y, n_x, n_spectral) — confirm
            # and build a (x, y, shift) DataArray with spatial coords.
            n_y, n_x, n_spec = data.shape
            if n_spec != shift_si.size:
                raise ValueError(
                    f"WDF spectral axis length ({shift_si.size}) does not match "
                    f"data.shape[-1] ({n_spec})"
                )
            da = xr.DataArray(
                data=data,
                dims=["y", "x", "shift"],
                coords={
                    "shift": shift_si,
                    "x": np.arange(n_x),
                    "y": np.arange(n_y),
                },
                attrs={
                    "values_label": "counts",
                    "shift_units": "m^-1",
                    "excitation_wavelength_nm": excitation_nm,
                    "WDFResult": str(wdf),  # store repr; object not serialisable
                },
                name="counts",
            )
            return cls(
                da=da,
                excitation_wavelength_nm=excitation_nm,
                metadata={"WDFResult": wdf},
                values_label="counts",
            )

        # ---- 1-D or 2-D (time series) ------------------------------------
        return cls.from_arrays(
            shift_per_cm=shift_per_cm,
            values=data,
            excitation_wavelength_nm=excitation_nm,
            values_label="counts",
            metadata={"WDFResult": wdf},
        )

    # ----------------------------------------------------------------
    # Dunder helpers
    # ----------------------------------------------------------------

    def __repr__(self) -> str:
        shift = self.shift_per_cm
        shift_range = (
            f"{shift.min():.0f}–{shift.max():.0f} cm⁻¹" if shift.size else "empty"
        )
        dims = dict(zip(self.da.dims, self.da.shape))
        return (
            f"RamanData("
            f"label={self.values_label!r}, "
            f"excitation={self.excitation_wavelength_nm:.1f} nm, "
            f"shape={dims}, "
            f"shift={shift_range}"
            f")"
        )

    def __len__(self) -> int:
        return self.da.sizes.get("scan", 1)

    # ----------------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _build_da(
        shift_si: npt.NDArray,
        values: npt.NDArray,
        values_label: str,
        tos: Optional[npt.NDArray],
        metadata: Optional[dict[str, Any]],
    ) -> xr.DataArray:
        """Build the canonical DataArray from validated raw arrays."""
        coords: dict[str, Any] = {"shift": shift_si}
        dims: list[str]

        if values.ndim == 1:
            dims = ["shift"]
        else:
            dims = ["scan", "shift"]
            coords["scan"] = np.arange(values.shape[0])
            if tos is not None:
                coords["tos"] = ("scan", np.asarray(tos, dtype=float))

        return xr.DataArray(
            data=values,
            coords=coords,
            dims=dims,
            attrs={
                "values_label": values_label,
                "shift_units": "m^-1",
                **(metadata or {}),
            },
            name=values_label,
        )
