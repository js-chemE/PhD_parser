
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
 
from phd_parser.infrared import omnic
from phd_parser.units import (
    transform_matching_dimensions,
    transform_wavenumber_frequency,
)
 
logger = logging.getLogger(__name__)  # caller configures level / handlers

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
XLabel = Literal["wavenumber", "frequency", "energy"]
VLabel = Literal["absorbance", "transmittance", "reflectance"]



class IRData(BaseModel): 
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        ignored_types=(cached_property,),  # keeps cached_property working alongside Pydantic
    )
 
    # ----------------------------------------------------------------
    # Fields
    # ----------------------------------------------------------------
 
    # Core data — stored in SI units (cm-1), normalised on ingestion
    da: xr.DataArray = Field(
        description=(
            "xarray DataArray with dims ('wavenumber',) or ('scan', 'wavenumber'). "
            "Wavenumber coordinate is in SI units (cm-1). "
            "Optional coords: 'tos' (seconds), 'timestamp' (datetime)."
        )
    )
 
    # Metadata
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
        if not isinstance(v, xr.DataArray):
            raise TypeError(f"'da' must be an xr.DataArray, got {type(v)}")
        if "wavenumber" not in v.dims:
            raise ValueError("DataArray must have a 'wavenumber' dimension")
        if v.ndim not in (1, 2):
            raise ValueError(f"DataArray must be 1-D or 2-D, got {v.ndim}-D")
        if v.ndim == 2 and v.dims[0] != "scan":
            raise ValueError("2-D DataArray must have dims ('scan', 'wavenumber')")
        return v
 
    @model_validator(mode="after")
    def validate_attrs(self) -> "IRData":
        self.da.attrs.setdefault("values_label", "absorbance")
        self.da.attrs.setdefault("wavenumber_units", "m^-1")
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
    def values_label(self) -> str:
        return self.da.attrs["values_label"]
 
    @property
    def wavenumber(self) -> npt.NDArray:
        """Wavenumber in SI units (m⁻¹)."""
        return self.da.coords["wavenumber"].values
 
    @property
    def wavenumber_per_cm(self) -> npt.NDArray:
        """Wavenumber in conventional cm⁻¹."""
        return self.wavenumber / 100.0
 
    @property
    def tos(self) -> Optional[npt.NDArray]:
        """Time-of-scan in seconds, or None."""
        if "tos" in self.da.coords:
            return self.da.coords["tos"].values
        return None
 
    @property
    def timestamps(self) -> Optional[pd.DatetimeIndex]:
        """Absolute timestamps per scan, or None."""
        if "timestamp" in self.da.coords:
            return pd.DatetimeIndex(self.da.coords["timestamp"].values)
        return None
    
    # ----------------------------------------------------------------
    # Cached unit-conversion properties
    # ----------------------------------------------------------------
 
    @cached_property
    def wavelength(self) -> npt.NDArray:
        """Wavelength in metres."""
        return 1.0 / self.wavenumber
 
    @cached_property
    def wavelength_nm(self) -> npt.NDArray:
        """Wavelength in nanometres."""
        return self.wavelength * 1e9
 
    @cached_property
    def frequency(self) -> npt.NDArray:
        """Frequency in Hz."""
        return self.wavenumber * const.c
 
    @cached_property
    def energy(self) -> npt.NDArray:
        """Photon energy in Joules."""
        return self.wavenumber * const.Planck * const.c
 
    @cached_property
    def energy_eV(self) -> npt.NDArray:
        """Photon energy in electronvolts."""
        return self.energy / const.electron_volt
    
    # ----------------------------------------------------------------
    # Spectral indexing
    # ----------------------------------------------------------------
 
    def get_scan(self, scan_index: int) -> npt.NDArray:
        if self.ndim == 1:
            raise ValueError("get_scan requires 2-D data")
        if not (0 <= scan_index < self.shape[0]):
            raise IndexError(
                f"scan_index {scan_index} out of bounds for {self.shape[0]} scans"
            )
        return self.da.isel(scan=scan_index).values
 
    def get_evolution(
        self,
        wavenumber_per_cm: Union[float, list[float], npt.NDArray],
        method: Literal["nearest", "linear"] = "nearest",
        tolerance_per_cm: Optional[float] = None,
    ) -> xr.DataArray:
        
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
 
        return self.da.sel(wavenumber=targets_si, method=method)
    
    # ----------------------------------------------------------------
    # Immutable transformations
    # ----------------------------------------------------------------
 
    def sort(self, ascending: bool = True) -> "IRData":
        da_sorted = self.da.sortby("wavenumber", ascending=ascending)
        return IRData(da=da_sorted, metadata=self.metadata)

    def select_wavenumber_range(
        self,
        min_cm: Optional[float] = None,
        max_cm: Optional[float] = None,
    ) -> "IRData":
        da = self.da

        if min_cm is not None:
            wn = da.coords["wavenumber"].values          # read from current da
            da = da.sel(wavenumber=wn >= min_cm * 100.0)

        if max_cm is not None:
            wn = da.coords["wavenumber"].values          # re-read after potential slice
            da = da.sel(wavenumber=wn <= max_cm * 100.0)

        da_selected = self._build_da(
            wavenumber_si=da.coords["wavenumber"].values,
            values=da.values,
            values_label=self.values_label,
            tos=da.coords["tos"].values if "tos" in da.coords else None,
            tos_start=None,
            metadata=self.metadata,
        )
        return IRData(da=da_selected, metadata=self.metadata)
    
    def select_tos_range(
        self,
        min_s: Optional[float] = None,
        max_s: Optional[float] = None,
    ) -> "IRData":
        da = self.da
        if min_s is not None:
            tos = da.coords.get("tos")
            da = da.sel(tos=tos >= min_s)
        if max_s is not None:
            tos = da.coords.get("tos")
            da = da.sel(tos=tos <= max_s)

        da_selected = self._build_da(
            wavenumber_si=self.wavenumber,
            values=da.values,
            values_label=self.values_label,
            tos=da.coords["tos"].values if "tos" in da.coords else None,
            tos_start=None,  # can't determine a single tos_start for a subset of scans
            metadata=self.metadata,
        )
        return IRData(da=da_selected, metadata=self.metadata)
    
    def smooth(self, method: Literal["moving", "savgol", "gaussian"], **kwargs: Any) -> "IRData":
        if method == "moving":
            return self.smooth_moving(**kwargs)
        elif method == "savgol":
            return self.smooth_savgol(**kwargs)
        elif method == "gaussian":
            return self.smooth_gaussian(**kwargs)
        else:
            raise ValueError(f"Unsupported smoothing method: {method}")
    
    def smooth_savgol(self, window_length: int = 21, polyorder: int = 3) -> "IRData":
        from scipy.signal import savgol_filter

        if self.ndim == 1:
            smoothed_values = savgol_filter(self.values, window_length, polyorder)
        else:
            smoothed_values = np.apply_along_axis(
                lambda m: savgol_filter(m, window_length, polyorder),
                axis=1,
                arr=self.values,
            )

        da_smoothed = self._build_da(
            wavenumber_si=self.wavenumber,
            values=smoothed_values,
            values_label=self.values_label + "_smoothed",
            tos=self.tos,
            tos_start=None,  # can't determine a single tos_start for smoothed scans
            metadata=self.metadata,
        )
        return IRData(da=da_smoothed, metadata=self.metadata)
    
    def smooth_gaussian(self, sigma_cm: float) -> "IRData":
        from scipy.ndimage import gaussian_filter1d

        sigma_si = sigma_cm * 100.0
        if self.ndim == 1:
            smoothed_values = gaussian_filter1d(self.values, sigma=sigma_si)
        else:
            smoothed_values = np.apply_along_axis(
                lambda m: gaussian_filter1d(m, sigma=sigma_si),
                axis=1,
                arr=self.values,
            )

        da_smoothed = self._build_da(
            wavenumber_si=self.wavenumber,
            values=smoothed_values,
            values_label=self.values_label + "_smoothed",
            tos=self.tos,
            tos_start=None,  # can't determine a single tos_start for smoothed scans
            metadata=self.metadata,
        )
        return IRData(da=da_smoothed, metadata=self.metadata)
        
    def smooth_moving(self, window_size: int = 5) -> "IRData":
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.ndim == 1:
            smoothed_values = np.convolve(
                self.values, np.ones(window_size) / window_size, mode="same"
            )
        else:
            smoothed_values = np.apply_along_axis(
                lambda m: np.convolve(m, np.ones(window_size) / window_size, mode="same"),
                axis=1,
                arr=self.values,
            )

        da_smoothed = self._build_da(
            wavenumber_si=self.wavenumber,
            values=smoothed_values,
            values_label=self.values_label + "_smoothed",
            tos=self.tos,
            tos_start=None,  # can't determine a single tos_start for smoothed scans
            metadata=self.metadata,
        )
        return IRData(da=da_smoothed, metadata=self.metadata)
    
    def average_scans(self, number_of_scans: int, tos_method: Literal["mean", "median", "first", "last"] = "first") -> "IRData":
        if self.ndim == 1:
            raise ValueError("average_scans requires 2-D data")
        if number_of_scans < 1:
            raise ValueError("number_of_scans must be >= 1")

        n_scans = self.shape[0]
        n_averaged = n_scans // number_of_scans
        new_values = self.values[:n_averaged * number_of_scans].reshape(
            n_averaged, number_of_scans, -1
        ).mean(axis=1)

        new_tos = None
        if self.tos is not None:
            new_tos = self.tos[:n_averaged * number_of_scans].reshape(
                n_averaged, number_of_scans
            )
            if tos_method == "mean":
                new_tos = new_tos.mean(axis=1)
            elif tos_method == "median":
                new_tos = np.median(new_tos, axis=1)
            elif tos_method == "first":
                new_tos = new_tos[:, 0]
            elif tos_method == "last":
                new_tos = new_tos[:, -1]

        da_averaged = self._build_da(
            wavenumber_si=self.wavenumber,
            values=new_values,
            values_label=self.values_label,
            tos=new_tos,
            tos_start=None,  # can't determine a single tos_start for averaged scans
            metadata=self.metadata,
        )

        return IRData(da=da_averaged, metadata=self.metadata)

    # ----------------------------------------------------------------
    # Export
    # ----------------------------------------------------------------
 
    def to_csv(
        self,
        filepath: Union[str, Path],
        wavenumber_units: Literal["cm-1", "m-1"] = "cm-1",
    ) -> None:
        filepath = Path(filepath)
        wn = self.wavenumber_per_cm if wavenumber_units == "cm-1" else self.wavenumber
        wn_label = f"wavenumber [{wavenumber_units}]"
 
        if self.ndim == 1:
            df = pd.DataFrame(
                {self.values_label: self.values},
                index=pd.Index(wn, name=wn_label),
            )
        else:
            tos = self.tos
            col_labels = (
                [f"tos_{t:.2f}s" for t in tos]
                if tos is not None
                else [f"scan_{i}" for i in range(self.shape[0])]
            )
            df = pd.DataFrame(
                self.values.T,
                index=pd.Index(wn, name=wn_label),
                columns=col_labels,
            )
 
        df.to_csv(filepath)
        logger.debug("Saved CSV → %s", filepath)
 
    def to_netcdf(self, filepath: Union[str, Path]) -> None:
        """
        Save to NetCDF4 via xarray. Preserves all coordinates and metadata.
 
        Reload with ``IRData.from_netcdf(filepath)``.
        """
        self.da.to_netcdf(filepath)
        logger.debug("Saved NetCDF → %s", filepath)

    # ----------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        wavenumber_per_cm: npt.NDArray,
        values: npt.NDArray,
        values_label: VLabel = "absorbance",
        tos: Optional[npt.NDArray] = None,
        tos_start: Optional[Union[pd.Timestamp, str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "IRData":
        wavenumber_si = np.asarray(wavenumber_per_cm, dtype=float) * 100.0
        values = np.asarray(values, dtype=float)
 
        if wavenumber_si.ndim != 1:
            raise ValueError("wavenumber_per_cm must be 1-D")
        if values.ndim == 1:
            if values.size != wavenumber_si.size:
                raise ValueError(
                    f"values size ({values.size}) != wavenumber size ({wavenumber_si.size})"
                )
        elif values.ndim == 2:
            n_scans, n_pts = values.shape
            if n_pts != wavenumber_si.size:
                raise ValueError(
                    f"values.shape[1] ({n_pts}) != wavenumber size ({wavenumber_si.size})"
                )
            if tos is not None:
                tos = np.asarray(tos, dtype=float)
                if tos.ndim != 1 or tos.size != n_scans:
                    raise ValueError(
                        f"tos size ({tos.size}) != values.shape[0] ({n_scans})"
                    )
        else:
            raise ValueError(f"values must be 1-D or 2-D, got shape {values.shape}")
 
        da = cls._build_da(wavenumber_si, values, values_label, tos, tos_start, metadata)
        return cls(da=da, metadata=metadata or {})
 
    @classmethod
    def from_netcdf(cls, filepath: Union[str, Path]) -> "IRData":
        da = xr.open_dataarray(filepath)
        return cls(da=da, metadata=dict(da.attrs))
 
    @classmethod
    def from_xarray(
        cls,
        da: xr.DataArray,
        values_label: VLabel = "absorbance",
        metadata: Optional[dict[str, Any]] = None,
    ) -> "IRData":
        da = da.copy()
        da.attrs.setdefault("values_label", values_label)
        da.attrs.setdefault("wavenumber_units", "m^-1")
        return cls(da=da, metadata=metadata or dict(da.attrs))

    @classmethod
    def from_omnic_spa(
        cls,
        filepath: Union[str, Path],
        values_label: VLabel = "absorbance",
        wavenumber_2SI_factor: float = 100.0,
        delta_time_seconds: Optional[float] = None,
        tos_start: Optional[Union[pd.Timestamp, str]] = None,
        strict_tos_start: bool = True,
    ) -> "IRData":
        
        if delta_time_seconds is not None and tos_start is not None:
            raise ValueError(
                "Specify either 'delta_time_seconds' or 'tos_start', not both."
            )
 
        raw = omnic.read_spa(
            filepath,
            delta_time_seconds=delta_time_seconds,
            tos_start=tos_start,
        )
 
        wavenumber_si = np.asarray(
            transform_matching_dimensions(
                raw["data"]["x"],
                from_2SI_factor=wavenumber_2SI_factor,
                to_2SI_factor=1,
            ),
            dtype=float,
        )
        values = np.asarray(raw["data"]["v"], dtype=float)
 
        raw_tos_start = raw["meta"].get("tos_start")
        parsed_tos_start: Optional[pd.Timestamp] = None
        
        if raw_tos_start is not None:
            try:
                parsed_tos_start = pd.Timestamp(raw_tos_start)
            except Exception as exc:
                if strict_tos_start:
                    raise ValueError(
                        f"Could not parse tos_start '{raw_tos_start}': {exc}"
                    ) from exc
                logger.warning(
                    "Ignoring unparseable tos_start '%s': %s", raw_tos_start, exc
                )
 
        da = cls._build_da(
            wavenumber_si,
            values,
            values_label,
            tos=raw["data"].get("tos"),
            tos_start=parsed_tos_start,
            metadata=raw["meta"],
        )
        return cls(da=da, metadata=raw["meta"])
    

    # ----------------------------------------------------------------
    # Dunder helpers
    # ----------------------------------------------------------------
 
    def __repr__(self) -> str:
        wn = self.wavenumber_per_cm
        wn_range = f"{wn.min():.1f}–{wn.max():.1f} cm⁻¹" if wn.size else "empty"
        dims = dict(zip(self.da.dims, self.da.shape))
        return (
            f"IRData("
            f"label={self.values_label!r}, "
            f"shape={dims}, "
            f"wavenumber={wn_range}"
            f")"
        )
 
    def __len__(self) -> int:
        return self.da.sizes.get("scan", 1)


    # ----------------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------------
 
    @staticmethod
    def _build_da(
        wavenumber_si: npt.NDArray,
        values: npt.NDArray,
        values_label: str,
        tos: Optional[npt.NDArray],
        tos_start: Optional[Union[pd.Timestamp, str]],
        metadata: Optional[dict[str, Any]],
    ) -> xr.DataArray:
        """Build the canonical DataArray from validated raw arrays."""
        coords: dict[str, Any] = {"wavenumber": wavenumber_si}
        dims: list[str]
 
        if values.ndim == 1:
            dims = ["wavenumber"]
        else:
            dims = ["scan", "wavenumber"]
            coords["scan"] = np.arange(values.shape[0])
 
            if tos is not None:
                tos = np.asarray(tos, dtype=float)
                coords["tos"] = ("scan", tos)
 
                if tos_start is not None:
                    ts = pd.Timestamp(tos_start)
                    timestamps = [ts + pd.Timedelta(seconds=float(t)) for t in tos]
                    coords["timestamp"] = ("scan", timestamps)
 
        return xr.DataArray(
            data=values,
            coords=coords,
            dims=dims,
            attrs={
                "values_label": values_label,
                "wavenumber_units": "m^-1",
                **(metadata or {}),
            },
            name=values_label,
        )