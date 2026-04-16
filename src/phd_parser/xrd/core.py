import logging
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class XRDData(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        ignored_types=(cached_property,),
    )

    # ----------------------------------------------------------------
    # Fields
    # ----------------------------------------------------------------

    # Core data — angle in degrees (2θ), consistent throughout
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
        if not isinstance(v, xr.DataArray):
            raise TypeError(f"'da' must be an xr.DataArray, got {type(v)}")
        if "angle" not in v.dims:
            raise ValueError("DataArray must have an 'angle' dimension")
        if v.ndim not in (1, 2):
            raise ValueError(f"DataArray must be 1-D or 2-D, got {v.ndim}-D")
        if v.ndim == 2 and v.dims[0] != "scan":
            raise ValueError("2-D DataArray must have dims ('scan', 'angle')")
        return v

    @model_validator(mode="after")
    def validate_attrs(self) -> "XRDData":
        self.da.attrs.setdefault("angle_units", "deg")
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
    def angle(self) -> npt.NDArray:
        """2θ angle in degrees."""
        return self.da.coords["angle"].values


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
    def angle_rad(self) -> npt.NDArray:
        """2θ angle in radians."""
        return np.deg2rad(self.angle)

    @cached_property
    def theta_deg(self) -> npt.NDArray:
        """θ (Bragg angle) in degrees."""
        return self.angle / 2.0

    @cached_property
    def theta_rad(self) -> npt.NDArray:
        """θ (Bragg angle) in radians."""
        return np.deg2rad(self.theta_deg)

    def d_spacing(self, wavelength_angstrom: float = 1.5406) -> npt.NDArray:
        """
        d-spacing in Ångströms via Bragg's law: d = λ / (2 sin θ).

        Parameters
        ----------
        wavelength_angstrom : float
            X-ray wavelength in Å. Defaults to Cu Kα₁ (1.5406 Å).
        """
        sin_theta = np.sin(self.theta_rad)
        # Avoid division by zero at angle=0
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(sin_theta > 0, wavelength_angstrom / (2.0 * sin_theta), np.nan)
        return d

    def q_vector(self, wavelength_angstrom: float = 1.5406) -> npt.NDArray:
        """
        Scattering vector magnitude Q in Å⁻¹: Q = 4π sin θ / λ.

        Parameters
        ----------
        wavelength_angstrom : float
            X-ray wavelength in Å. Defaults to Cu Kα₁ (1.5406 Å).
        """
        return (4.0 * np.pi * np.sin(self.theta_rad)) / wavelength_angstrom

    # ----------------------------------------------------------------
    # Immutable transformations
    # ----------------------------------------------------------------

    def sort(self, ascending: bool = True) -> "XRDData":
        """Return a new XRDData with angles sorted."""
        da_sorted = self.da.sortby("angle", ascending=ascending)
        return XRDData(da=da_sorted, metadata=self.metadata)

    def select_angle_range(
        self,
        min_deg: Optional[float] = None,
        max_deg: Optional[float] = None,
    ) -> "XRDData":
        """Slice to a 2θ window [min_deg, max_deg]."""
        mask = np.ones(self.angle.size, dtype=bool)
        if min_deg is not None:
            mask &= self.angle >= min_deg
        if max_deg is not None:
            mask &= self.angle <= max_deg

        da = self.da.isel(angle=mask)
        return XRDData(
            da=self._build_da(
                angle=da.coords["angle"].values,
                values=da.values,
            ),
            metadata=self.metadata,
        )

    def smooth_savgol(
        self, window_length: int = 11, polyorder: int = 3
    ) -> "XRDData":
        from scipy.signal import savgol_filter

        if self.ndim == 1:
            smoothed = savgol_filter(self.values, window_length, polyorder)
        else:
            smoothed = np.apply_along_axis(
                lambda m: savgol_filter(m, window_length, polyorder),
                axis=1,
                arr=self.values,
            )
        da_s = self._build_da(
            angle=self.angle,
            values=smoothed,
        )
        return XRDData(da=da_s, metadata=self.metadata)

    def smooth_gaussian(self, sigma_deg: float = 0.05) -> "XRDData":
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
        da_s = self._build_da(
            angle=self.angle,
            values=smoothed,
        )
        return XRDData(da=da_s, metadata=self.metadata)

    def smooth_moving(self, window_size: int = 5) -> "XRDData":
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
        da_s = self._build_da(
            angle=self.angle,
            values=smoothed,
        )
        return XRDData(da=da_s, metadata=self.metadata)


    # ----------------------------------------------------------------
    # Export
    # ----------------------------------------------------------------


    # ----------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------

    @classmethod
    def from_e1290(
        cls,
        filepath: Union[str, Path],
        normalize: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "XRDData":
        """
        Read a Bruker/E1290-style two-column XY file.

        Parameters
        ----------
        filepath : path-like
        normalize : bool
            If True, divide intensity by its maximum on load.
        metadata : dict, optional
            Additional metadata to attach.
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