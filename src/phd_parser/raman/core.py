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
        return self.da.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.da.shape)

    @property
    def values(self) -> npt.NDArray:
        return self.da.values

    @property
    def n_spectral(self) -> int:
        """Number of spectral points."""
        return self.da.sizes["shift"]

    @property
    def shift(self) -> npt.NDArray:
        """Raman shift in SI units (m⁻¹), Stokes > 0."""
        return self.da.coords["shift"].values

    @property
    def shift_per_cm(self) -> npt.NDArray:
        """Raman shift in conventional cm⁻¹, Stokes > 0."""
        return self.shift / 100.0

    @property
    def tos(self) -> Optional[npt.NDArray]:
        """Time-of-scan in seconds, or None."""
        if "tos" in self.da.coords:
            return self.da.coords["tos"].values
        return None

    # ----------------------------------------------------------------
    # Cached derived spectral axes
    # ----------------------------------------------------------------

    @cached_property
    def excitation_wavenumber_per_cm(self) -> float:
        """Excitation laser wavenumber in cm⁻¹."""
        return 1e7 / self.excitation_wavelength_nm

    @cached_property
    def excitation_wavenumber(self) -> float:
        """Excitation laser wavenumber in m⁻¹."""
        return self.excitation_wavenumber_per_cm * 100.0

    @cached_property
    def wavenumber(self) -> npt.NDArray:
        """
        Absolute scattered wavenumber in SI units (m⁻¹).

        For Stokes: wavenumber_scattered = wavenumber_excitation - shift
        """
        return self.excitation_wavenumber - self.shift

    @cached_property
    def wavenumber_per_cm(self) -> npt.NDArray:
        """Absolute scattered wavenumber in cm⁻¹."""
        return self.wavenumber / 100.0

    @cached_property
    def wavelength(self) -> npt.NDArray:
        """Scattered wavelength in metres."""
        return 1.0 / self.wavenumber

    @cached_property
    def wavelength_nm(self) -> npt.NDArray:
        """Scattered wavelength in nanometres."""
        return self.wavelength * 1e9

    @cached_property
    def frequency(self) -> npt.NDArray:
        """Scattered photon frequency in Hz."""
        return self.wavenumber * const.c

    # ----------------------------------------------------------------
    # Spectral indexing
    # ----------------------------------------------------------------

    def get_scan(self, scan_index: int) -> npt.NDArray:
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
        if "x" not in self.da.dims or "y" not in self.da.dims:
            raise ValueError("get_map_spectrum requires 'x' and 'y' dimensions")
        return self.da.isel(x=x, y=y).values

    # ----------------------------------------------------------------
    # immutable transformations
    # ----------------------------------------------------------------

    def sort(self, ascending: bool = True) -> "RamanData":
        """
        Return a new RamanData with the shift axis sorted.

        The original object is not modified. Immutable style is used
        because cached_property values are tied to the instance.
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
        """
        Return a new RamanData with only the points within the specified shift range.

        The original object is not modified. Immutable style is used
        because cached_property values are tied to the instance.
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
        """Read a BTC655N spectrometer export file."""
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