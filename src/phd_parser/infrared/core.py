import logging
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union, List, Tuple

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
}


class IRData(BaseModel):
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

    @model_validator(mode="after")
    def validate_attrs(self) -> "IRData":
        return self

    # ----------------------------------------------------------------
    # Core properties
    # ----------------------------------------------------------------

    @property
    def ndim(self) -> int:
        return self.ds["data"].ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.ds["data"].shape)

    @property
    def values(self) -> npt.NDArray:
        return self.ds["data"].values

    @property
    def wavenumber(self) -> npt.NDArray:
        # SI units (m⁻¹)
        return self.ds.coords["wavenumber"].values

    @property
    def wavenumber_per_cm(self) -> npt.NDArray:
        return self.wavenumber / 100.0

    @property
    def tos(self) -> Optional[npt.NDArray]:
        # Elapsed seconds since first scan
        if "tos" in self.ds.coords:
            return self.ds.coords["tos"].values
        return None

    @property
    def tos_start(self) -> Optional[pd.Timestamp]:
        # Parse from attributes; not stored as a coordinate since it's a single value applying to all scans
        raw = self.ds.attrs.get("tos_start")
        if raw is None:
            return None
        return pd.Timestamp(raw)

    @property
    def data_type(self) -> Optional[IRDataType]:
        return self.ds.attrs.get("data_type")

    @property
    def has_background(self) -> bool:
        return "background" in self.ds

    @property
    def background(self) -> Optional[npt.NDArray]:
        if "background" not in self.ds:
            return None
        return self.ds["background"].values

    @property
    def background_data_type(self) -> Optional[IRDataType]:
        if "background" not in self.ds:
            return None
        return self.ds["background"].attrs.get("data_type")

    @property
    def timestamps(self) -> Optional[pd.DatetimeIndex]:
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
        # metres
        return 1.0 / self.wavenumber

    @cached_property
    def wavelength_nm(self) -> npt.NDArray:
        return self.wavelength * 1e9
    
    @cached_property
    def wavelength_mum(self) -> npt.NDArray:
        return self.wavelength * 1e6

    @cached_property
    def frequency(self) -> npt.NDArray:
        # Hz
        return self.wavenumber * const.c

    @cached_property
    def energy(self) -> npt.NDArray:
        # Joules
        return self.wavenumber * const.Planck * const.c

    @cached_property
    def energy_eV(self) -> npt.NDArray:
        return self.energy / const.electron_volt
    
    @cached_property
    def energy_kJ_per_mol(self) -> npt.NDArray:
        return 1e-3 * self.energy * const.Avogadro # kJ/mol

    # ----------------------------------------------------------------
    # Get
    # ----------------------------------------------------------------

    def get_scan(self, scan_index: int) -> npt.NDArray:
        if self.ndim == 1:
            raise ValueError("get_scan requires 2-D data")
        if not (0 <= scan_index < self.shape[0]):
            raise IndexError(
                f"scan_index {scan_index} out of bounds for {self.shape[0]} scans"
            )
        return self.ds["data"].isel(scan=scan_index).values
    
    def get_scan_by_tos(
        self,
        target_tos: Union[float, Sequence[float]],
        method: Literal["nearest", "linear"] = "nearest",
        tolerance_seconds: Optional[float] = 10,
    ) -> Union[npt.NDArray]:
        if self.ndim == 1:
            raise ValueError("get_scan_by_tos requires 2-D data")
        if self.tos is None:
            raise ValueError("get_scan_by_tos requires 'tos' coordinate")

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
            return self.ds["data"].sel(tos=t, method=method).values

        results = np.vstack([_fetch_one(t) for t in targets])
        return results[0] if scalar_input else results


    def get_scan_by_tos_average(
        self,
        target_tos: Union[float, Sequence[float]],
        method: Literal["nearest", "linear"] = "nearest",
        tolerance_seconds: Optional[float] = 10,
        number_of_scans: Optional[int] = None,
        time_window: Optional[float] = None,
        direction: Literal["forward", "backward", "center"] = "center",
    ) -> Union[npt.NDArray]:
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

        bg_attrs: dict[str, Any] = {}
        if bg_data_type is not None:
            bg_attrs["data_type"] = bg_data_type

        bg_da = xr.DataArray(
            data=bg_values,
            coords={"wavenumber": self.ds.coords["wavenumber"]},
            dims=["wavenumber"],
            name="background",
            attrs=bg_attrs,
        )
        return IRData(ds=self.ds.assign({"background": bg_da}))

    # ----------------------------------------------------------------
    # Immutable — selection and sorting
    # ----------------------------------------------------------------

    def assign_tos_start(self, tos_start: Union[pd.Timestamp, str]) -> "IRData":
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
        return IRData(ds=self._carry_background(ds))

    def sort(self, by: str | Sequence[str] = "wavenumber", ascending: bool = True) -> "IRData":
        ds_sorted = self.ds.sortby(by, ascending=ascending)
        return IRData(ds=ds_sorted)
    
    def select_by_idx(self, idx: int) -> "IRData":
        if self.ndim == 1:
            raise ValueError("select_by_idx requires 2-D data")
        if not (0 <= idx < self.shape[0]):
            raise IndexError(f"idx {idx} out of bounds for {self.shape[0]} scans")
        ds_selected = self.ds.isel(scan=idx)
        return IRData(ds=ds_selected)

    def select_by_tos(self, target_tos: float, method: Literal["nearest", "linear"] = "nearest", tolerance_seconds: Optional[float] = 10) -> "IRData":
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
        return IRData(ds=self._slice_background_to(ds_new))

    def select_tos_range(
        self,
        min_s: Optional[float] = None,
        max_s: Optional[float] = None,
    ) -> "IRData":
        if self.tos is None:
            raise ValueError("select_tos_range requires a 'tos' coordinate")

        da = self.ds["data"]
        if min_s is not None:
            tos = da.coords["tos"].values
            if not np.any(tos >= min_s):
                min_s = tos[0]
                logger.warning(f"min_s {min_s:.1f}s is greater than all 'tos' values; using min_s={min_s:.1f}s instead")
            da = da.isel(scan=tos >= min_s)
        if max_s is not None:
            tos = da.coords["tos"].values
            if not np.any(tos <= max_s):
                max_s = tos[-1]
                logger.warning(f"max_s {max_s:.1f}s is less than all 'tos' values; using max_s={max_s:.1f}s instead")
            da = da.isel(scan=tos <= max_s)

        # tos values are absolute elapsed seconds, so tos_start + tos[i] remains valid
        ds_new = self._build_ds(
            wavenumber_si=da.coords["wavenumber"].values,
            values=da.values,
            tos=da.coords["tos"].values,
            attrs=dict(self.ds.attrs),
        )
        return IRData(ds=self._carry_background(ds_new))

    # ----------------------------------------------------------------
    # Immutable — smoothing
    # ----------------------------------------------------------------

    def smooth_savgol(self, window_length: int = 21, polyorder: int = 3) -> "IRData":
        from scipy.signal import savgol_filter

        if self.ndim == 1:
            smoothed = savgol_filter(self.values, window_length, polyorder)
        else:
            smoothed = np.apply_along_axis(
                lambda m: savgol_filter(m, window_length, polyorder), axis=1, arr=self.values
            )
        ds_new = self._build_ds(self.wavenumber, smoothed, tos=self.tos, attrs=dict(self.ds.attrs))
        return IRData(ds=self._carry_background(ds_new))

    def smooth_gaussian(self, sigma_cm: float) -> "IRData":
        from scipy.ndimage import gaussian_filter1d

        sigma_si = sigma_cm * 100.0
        if self.ndim == 1:
            smoothed = gaussian_filter1d(self.values, sigma=sigma_si)
        else:
            smoothed = np.apply_along_axis(
                lambda m: gaussian_filter1d(m, sigma=sigma_si), axis=1, arr=self.values
            )
        ds_new = self._build_ds(self.wavenumber, smoothed, tos=self.tos, attrs=dict(self.ds.attrs))
        return IRData(ds=self._carry_background(ds_new))

    def smooth_moving(self, window_size: int = 5) -> "IRData":
        if window_size < 1:
            raise ValueError("window_size must be >= 1")

        kernel = np.ones(window_size) / window_size
        if self.ndim == 1:
            smoothed = np.convolve(self.values, kernel, mode="same")
        else:
            smoothed = np.apply_along_axis(
                lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=self.values
            )
        ds_new = self._build_ds(self.wavenumber, smoothed, tos=self.tos, attrs=dict(self.ds.attrs))
        return IRData(ds=self._carry_background(ds_new))

    # ----------------------------------------------------------------
    # Immutable — baseline correction
    # ----------------------------------------------------------------

    def correct_offset(
        self,
        anchor_range_cm: Tuple[float, float] = (2500, 2600),
    ) -> "IRData":
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
            corrected = self.values - self.values[mask].mean()
        else:
            corrected = self.values - self.values[:, mask].mean(axis=1, keepdims=True)

        new_attrs = {**self.ds.attrs, "baseline_anchor_range_cm": list(anchor_range_cm)}
        ds_new = self._build_ds(wn, corrected, tos=self.tos, attrs=new_attrs)
        return IRData(ds=self._carry_background(ds_new))

    def correct_pchip(
        self,
        control_points_cm: Sequence[float],
        point_avg_half_width: int = 0,
    ) -> "IRData":
        from scipy.interpolate import PchipInterpolator

        wn_si = self.wavenumber
        wn_cm = wn_si / 100.0
        cps = np.sort(np.asarray(control_points_cm, dtype=float))

        def _subtract_pchip(spectrum_1d: np.ndarray) -> np.ndarray:
            x_knots = np.empty(len(cps))
            y_knots = np.empty(len(cps))
            for j, cp_cm in enumerate(cps):
                idx = int(np.abs(wn_cm - cp_cm).argmin())
                lo = max(0, idx - point_avg_half_width)
                hi = min(len(spectrum_1d), idx + point_avg_half_width + 1)
                x_knots[j] = wn_cm[idx]
                y_knots[j] = spectrum_1d[lo:hi].mean()
            return spectrum_1d - PchipInterpolator(x_knots, y_knots)(wn_cm)

        if self.ndim == 1:
            corrected = _subtract_pchip(self.values)
        else:
            corrected = np.apply_along_axis(_subtract_pchip, axis=1, arr=self.values)

        new_attrs = {
            **self.ds.attrs,
            "baseline_pchip_control_points_cm": sorted(control_points_cm),
            "baseline_pchip_half_width": point_avg_half_width,
        }
        ds_new = self._build_ds(wn_si, corrected, tos=self.tos, attrs=new_attrs)
        return IRData(ds=self._carry_background(ds_new))

    def correct_baseline(
        self,
        anchor_range_cm: Tuple[float, float] = (2500, 2600),
        control_points_cm: Optional[Sequence[float]] = None,
        point_avg_half_width: int = 0,
        double_offset: bool = True,
    ) -> "IRData":
        # Step 1: offset, step 2: PCHIP, step 3: optional second offset (mirrors DRIFTS behaviour)
        result = self.correct_offset(anchor_range_cm)
        if control_points_cm:
            result = result.correct_pchip(control_points_cm, point_avg_half_width)
            if double_offset:
                result = result.correct_offset(anchor_range_cm)
        return result

    def reapply_baseline(self) -> "IRData":
        # Re-runs correction using parameters stored in attributes (e.g. after average_scans)
        anchor_range_cm = self.ds.attrs.get("baseline_anchor_range_cm")
        if anchor_range_cm is None:
            raise ValueError("No baseline parameters found in attributes.")
        return self.correct_baseline(
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
        return IRData(ds=self._carry_background(ds_new))

    # ----------------------------------------------------------------
    # Immutable - Normalisation
    # ----------------------------------------------------------------

    def normalise_max(self) -> "IRData":
        max_val = self.values.max()
        if max_val == 0:
            logger.warning("Maximum value is zero; returning original data without normalisation")
            return self
        new_values = self.values / max_val
        ds_new = self._build_ds(self.wavenumber, new_values, tos=self.tos, attrs=dict(self.ds.attrs))
        return IRData(ds=self._carry_background(ds_new))

    def normalise_integral(self) -> "IRData":
        integral = np.trapz(self.values, x=self.wavenumber, axis=-1)
        if np.any(integral == 0):
            logger.warning("Integral is zero for some scans; returning original data without normalisation")
            return self
        new_values = self.values / integral[..., np.newaxis]
        ds_new = self._build_ds(self.wavenumber, new_values, tos=self.tos, attrs=dict(self.ds.attrs))
        return IRData(ds=self._carry_background(ds_new))

    def normalise_reference(self, reference: npt.NDArray) -> "IRData":
        if reference.ndim != 1:
            raise ValueError("Reference spectrum must be 1-D")
        if reference.size != self.wavenumber.size:
            raise ValueError(f"Reference size ({reference.size}) does not match wavenumber size ({self.wavenumber.size})")
        if np.any(reference == 0):
            logger.warning("Reference spectrum contains zero values; returning original data without normalisation")
            return self

        new_values = self.values / reference
        ds_new = self._build_ds(self.wavenumber, new_values, tos=self.tos, attrs=dict(self.ds.attrs))
        return IRData(ds=self._carry_background(ds_new))

    def normalise_reference_scan(self, scan_index: int) -> "IRData":
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
        if self.ndim == 1:
            raise ValueError("normalise_reference_by_tos requires 2-D data")
        reference = self.get_scan_by_tos(target_tos, method=method, tolerance_seconds=tolerance_seconds)
        return self.normalise_reference(reference)

    def normalise_value_range(self, new_min: float = 0.0, new_max: float = 1.0) -> "IRData":
        old_min = self.values.min()
        old_max = self.values.max()
        if old_max == old_min:
            logger.warning("All values are the same; returning original data without normalisation")
            return self
        new_values = (self.values - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
        ds_new = self._build_ds(self.wavenumber, new_values, tos=self.tos, attrs=dict(self.ds.attrs))
        return IRData(ds=self._carry_background(ds_new))

    def normalise_value(self, factor: float) -> "IRData":
        if factor == 0:
            logger.warning("Normalisation factor is zero; returning original data without normalisation")
            return self
        new_values = self.values / factor
        ds_new = self._build_ds(self.wavenumber, new_values, tos=self.tos, attrs=dict(self.ds.attrs))
        return IRData(ds=self._carry_background(ds_new))

    # ----------------------------------------------------------------
    # Gram-Schmidt reconstructed chromatogram
    # ----------------------------------------------------------------

    def get_gram_schmidt(self, reference: npt.NDArray) -> xr.DataArray:
        """Gram-Schmidt vector: L2 norm of each scan's component orthogonal to the reference subspace."""
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
        with xr.open_dataset(filepath) as ds:
            ds = ds.copy()
        return cls(ds=ds)

    @classmethod
    def from_xarray(
        cls,
        da: Union[xr.DataArray, xr.Dataset],
    ) -> "IRData":
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
    ) -> "IRData":
        if delta_time_seconds is not None and tos_start is not None:
            raise ValueError("Specify either 'delta_time_seconds' or 'tos_start', not both.")

        raw = omnic.read_spa(filepath, delta_time_seconds=delta_time_seconds, tos_start=tos_start)

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
        return f"IRData(data_type={self.data_type}, shape={self.shape}, wavenumber={wn_range}{tos_info}{ts_info}{bg_info})"

    def __len__(self) -> int:
        return self.ds.sizes.get("scan", 1)
    
    def __add__(self, other: "IRData") -> "IRData":
        if not isinstance(other, IRData):
            return NotImplemented
        self._check_compatible(other, "add")
        new_attrs = {**self.ds.attrs, **other.ds.attrs}
        ds_new = self._build_ds(self.wavenumber, self.values + other.values, tos=self.tos, attrs=new_attrs)
        return IRData(ds=self._carry_background(ds_new))

    def __sub__(self, other: "IRData") -> "IRData":
        if not isinstance(other, IRData):
            return NotImplemented
        self._check_compatible(other, "subtract")
        new_attrs = {**self.ds.attrs, **other.ds.attrs}
        ds_new = self._build_ds(self.wavenumber, self.values - other.values, tos=self.tos, attrs=new_attrs)
        return IRData(ds=self._carry_background(ds_new))


    # ----------------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------------

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