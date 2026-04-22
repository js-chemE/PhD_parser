from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Optional, Union, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import phd_parser.massspec.quadstar as quadstar

logger = logging.getLogger(__name__)

BlockUnit = Literal["A", "mbar", "arbitrary"]

class MSData(BaseModel):

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        ignored_types=(cached_property,),
    )

    # ----------------------------------------------------------------
    # Fields
    # ----------------------------------------------------------------

    ds: xr.Dataset = Field(
        description=(
            "xr.Dataset with one DataArray per datablock named 'block_{id}'. "
            "Each DataArray has dims ('time', 'mz'). "
            "Time coord is elapsed seconds (SI). "
            "Optional coords: 'timestamp' (datetime64), 'cycle' (int)."
        )
    )
    block_meta: dict[int, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-block metadata keyed by block ID (type, unit, channels).",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="File-level metadata (filename, date, cycle counts, etc.).",
    )

    # ----------------------------------------------------------------
    # Validators
    # ----------------------------------------------------------------

    @field_validator("ds", mode="before")
    @classmethod
    def validate_dataset(cls, v: Any) -> xr.Dataset:
        if not isinstance(v, xr.Dataset):
            raise TypeError(f"'ds' must be an xr.Dataset, got {type(v)}")
        for name, da in v.data_vars.items():
            if "time" not in da.dims or "mz" not in da.dims:
                raise ValueError(
                    f"DataArray '{name}' must have dims ('time', 'mz'), got {da.dims}"
                )
        return v

    @model_validator(mode="after")
    def validate_block_meta_keys(self) -> "MSData":
        for name in self.ds.data_vars:
            block_id = int(name.split("_", 1)[1])
            if block_id not in self.block_meta:
                logger.warning(
                    "block_meta missing entry for block %d — unit info unavailable",
                    block_id,
                )
        return self

    # ----------------------------------------------------------------
    # Core properties
    # ----------------------------------------------------------------

    @property
    def block_ids(self) -> list[int]:
        return sorted(int(n.split("_", 1)[1]) for n in self.ds.data_vars)

    @property
    def n_blocks(self) -> int:
        return len(self.block_ids)

    @property
    def time(self) -> npt.NDArray:
        """Elapsed time in seconds (SI)."""
        return self.ds.coords["time"].values

    @property
    def n_time(self) -> int:
        return self.time.size

    @property
    def timestamps(self) -> Optional[pd.DatetimeIndex]:
        if "timestamp" in self.ds.coords:
            return pd.DatetimeIndex(self.ds.coords["timestamp"].values)
        return None
    
    @property
    def tos_start(self) -> Optional[pd.Timestamp]:
        """Timestamp corresponding to time=0, if available."""
        for da in self.ds.data_vars.values():
            if "tos_start" in da.attrs:
                return pd.to_datetime(da.attrs["tos_start"])
        return None
    
    @property
    def tos(self) -> Optional[npt.NDArray]:
        """Time on stream in seconds, if tos_start is available."""
        timestamps = self.timestamps
        tos_start = self.tos_start
        if tos_start is not None and timestamps is not None:
            return (timestamps - tos_start).total_seconds().values
        return None

    @property
    def cycle(self) -> Optional[npt.NDArray]:
        if "cycle" in self.ds.coords:
            return self.ds.coords["cycle"].values
        return None

    def _block(self, block_id: int) -> xr.DataArray:
        name = f"block_{block_id}"
        if name not in self.ds:
            raise KeyError(f"Block {block_id} not found. Available: {self.block_ids}")
        return self.ds[name]

    def mz(self, block_id: int = 0) -> npt.NDArray:
        return self._block(block_id).coords["mz"].values

    def values(self, block_id: int = 0) -> npt.NDArray:
        return self._block(block_id).values

    def unit(self, block_id: int = 0) -> str:
        return self.block_meta.get(block_id, {}).get("unit", "?")

    def block_type(self, block_id: int = 0) -> str:
        return self.block_meta.get(block_id, {}).get("type", "?")

    # ----------------------------------------------------------------
    # Cached derived quantities
    # ----------------------------------------------------------------

    @cached_property
    def _tic_cache(self) -> dict[int, npt.NDArray]:
        return {
            bid: np.nansum(self._block(bid).values, axis=1)
            for bid in self.block_ids
        }

    def tic(self, block_id: int = 0) -> npt.NDArray:
        """Total ion current vs time — sum over all m/z channels, NaN-safe."""
        return self._tic_cache[block_id]

    # ----------------------------------------------------------------
    # Trace / spectrum extraction
    # ----------------------------------------------------------------

    def get_trace(
        self,
        mz: float,
        block_id: int = 0,
        method: Literal["nearest", "linear"] = "nearest",
        tolerance: Optional[float] = 0.2,
    ) -> xr.DataArray:
        """Intensity vs time for a single m/z. Shape: (n_time,)."""
        da = self._block(block_id)
        if tolerance is not None:
            nearest_dist = float(np.abs(da.coords["mz"].values - mz).min())
            if nearest_dist > tolerance:
                logger.warning(
                    f"Requested m/z {mz} is {nearest_dist:.3f} Da from the nearest grid point (tolerance: {tolerance})"
                )
                raise ValueError(
                    f"Requested m/z {mz} is {nearest_dist:.3f} Da from the "
                    f"nearest grid point (tolerance: {tolerance})"
                )
        return da.sel(mz=mz, method=method)

    def get_traces(
        self,
        mz_list: list[float],
        block_id: int = 0,
        method: Literal["nearest", "linear"] = "nearest",
        tolerance: Optional[float] = None,
    ) -> xr.DataArray:
        """Intensity vs time for multiple m/z values. Shape: (n_time, n_mz)."""
        targets = np.atleast_1d(np.asarray(mz_list, dtype=float))
        da = self._block(block_id)
        if tolerance is not None:
            for t in targets:
                nearest_dist = float(np.abs(da.coords["mz"].values - t).min())
                if nearest_dist > tolerance:
                    logger.warning(
                        f"Requested m/z {t} is {nearest_dist:.3f} Da from the nearest grid point (tolerance: {tolerance})"
                    )
                    raise ValueError(
                        f"Requested m/z {t} is {nearest_dist:.3f} Da from the "
                        f"nearest grid point (tolerance: {tolerance})"
                    )
        return da.sel(mz=targets, method=method)

    def get_spectrum(self, time: float, block_id: int = 0) -> xr.DataArray:
        """Full mass spectrum at a single timepoint (nearest). Shape: (n_mz,)."""
        return self._block(block_id).sel(time=time, method="nearest")

    # ----------------------------------------------------------------
    # Immutable slicing
    # ----------------------------------------------------------------

    def select_tos_range(
        self,
        tos_start_seconds: Optional[float] = None,
        tos_end_seconds: Optional[float] = None,
    ) -> "MSData":
        """Return a new MSData restricted to [tos_start_seconds, tos_end_seconds] seconds."""
        tos = self.tos
        if tos is None:
            raise ValueError("Cannot select by time on stream because tos_start or timestamps are not available.")
        
        mask = np.ones(tos.size, dtype=bool)
        if tos_start_seconds is not None:
            mask &= tos >= tos_start_seconds
        if tos_end_seconds is not None:
            mask &= tos <= tos_end_seconds
        return MSData(
            ds=self.ds.isel(time=mask),
            block_meta=self.block_meta,
            metadata=self.metadata,
        )
    
    # ----------------------------------------------------------------
    # Immutable Baseline Correction
    # ----------------------------------------------------------------

    def correct_traces(
        self,
        mz: Union[None, Literal["all"], float, Sequence[float]] = "all",
        block_id: int = 0,
        tolerance: Optional[float] = 0.2,
    ) -> "MSData":
        # No-op
        if mz is None:
            return MSData(
                ds=self.ds.copy(deep=True),
                block_meta=self.block_meta,
                metadata=self.metadata,
            )

        new_ds = self.ds.copy(deep=True)
        shifts_log: dict[int, dict[float, float]] = {}

        # "all": every channel in every block
        if isinstance(mz, str):
            if mz != "all":
                raise ValueError(f"mz string argument must be 'all', got {mz!r}")
            for bid in self.block_ids:
                name = f"block_{bid}"
                arr = new_ds[name].values
                mins = np.nanmin(arr, axis=0)
                shift = np.where(mins < 0, mins, 0.0)
                new_ds[name] = new_ds[name] - shift
                # Log only channels that actually moved
                mz_grid = new_ds[name].coords["mz"].values
                moved = {float(mz_grid[i]): float(-shift[i]) for i in np.where(shift < 0)[0]}
                if moved:
                    shifts_log[bid] = moved

        # float or sequence of floats: targeted
        else:
            try:
                targets = np.atleast_1d(np.asarray(mz, dtype=float))
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"mz must be None, 'all', a float, or a sequence of floats; got {mz!r}"
                ) from e

            name = f"block_{block_id}"
            if name not in new_ds:
                raise KeyError(f"Block {block_id} not found. Available: {self.block_ids}")

            mz_grid = new_ds[name].coords["mz"].values
            arr = new_ds[name].values
            block_log: dict[float, float] = {}

            for target in targets:
                dists = np.abs(mz_grid - target)
                idx = int(np.argmin(dists))
                if tolerance is not None and dists[idx] > tolerance:
                    raise ValueError(
                        f"Requested m/z {target} is {dists[idx]:.3f} Da from the "
                        f"nearest grid point (tolerance: {tolerance})"
                    )
                trace_min = np.nanmin(arr[:, idx])
                if trace_min < 0:
                    arr[:, idx] = arr[:, idx] - trace_min
                    block_log[float(mz_grid[idx])] = float(-trace_min)

            new_ds[name].values[...] = arr
            if block_log:
                shifts_log[block_id] = block_log

        # Audit trail
        new_metadata = dict(self.metadata)
        if shifts_log:
            history = new_metadata.get("trace_corrections", [])
            history = list(history) + [{"method": "min_shift", "shifts": shifts_log}]
            new_metadata["trace_corrections"] = history

        return MSData(ds=new_ds, block_meta=self.block_meta, metadata=new_metadata)


    def baseline_subtract(
        self,
        tos_start_seconds: float,
        tos_end_seconds: float,
        block_id: Union[int, Literal["all"], None] = "all",
    ) -> "MSData":
        # Compute per-channel mean over the baseline window (requires tos)
        tos = self.tos
        if tos is None:
            raise ValueError(
                "Cannot baseline-subtract by tos range because tos_start or "
                "timestamps are not available."
            )

        mask = (tos >= tos_start_seconds) & (tos <= tos_end_seconds)
        if not mask.any():
            raise ValueError(
                f"Baseline window [{tos_start_seconds}, {tos_end_seconds}] s "
                f"contains no samples. tos range is "
                f"[{float(tos.min()):.1f}, {float(tos.max()):.1f}] s."
            )

        # Resolve which blocks to process
        if block_id is None or block_id == "all":
            target_blocks = self.block_ids
        else:
            if block_id not in self.block_ids:
                raise KeyError(f"Block {block_id} not found. Available: {self.block_ids}")
            target_blocks = [block_id]

        new_ds = self.ds.copy(deep=True)
        baseline_log: dict[int, dict[str, Any]] = {}

        for bid in target_blocks:
            name = f"block_{bid}"
            da = new_ds[name]
            # Per-channel mean over the masked time window
            baseline = da.isel(time=mask).mean(dim="time", skipna=True)
            new_ds[name] = da - baseline
            baseline_log[bid] = {
                "mean_per_mz": {
                    float(m): float(b)
                    for m, b in zip(da.coords["mz"].values, baseline.values)
                },
                "n_samples": int(mask.sum()),
            }

        new_metadata = dict(self.metadata)
        history = new_metadata.get("trace_corrections", [])
        history = list(history) + [
            {
                "method": "baseline_subtract",
                "window_tos_s": [float(tos_start_seconds), float(tos_end_seconds)],
                "baselines": baseline_log,
            }
        ]
        new_metadata["trace_corrections"] = history

        return MSData(ds=new_ds, block_meta=self.block_meta, metadata=new_metadata)

    # ----------------------------------------------------------------
    # Export
    # ----------------------------------------------------------------

    def to_csv(self, filepath: Union[str, Path], block_id: int = 0) -> None:
        filepath = Path(filepath)
        da = self._block(block_id)
        df = pd.DataFrame(
            da.values,
            index=pd.Index(self.time, name="time [s]"),
            columns=[f"mz_{m}" for m in da.coords["mz"].values],
        )
        if self.timestamps is not None:
            df.insert(0, "timestamp", self.timestamps)
        df.to_csv(filepath)
        logger.debug("Saved block %d → %s", block_id, filepath)

    def to_netcdf(self, filepath: Union[str, Path]) -> None:
        """Save full dataset to NetCDF4. Reload with from_netcdf()."""
        self.ds.to_netcdf(filepath)
        logger.debug("Saved NetCDF → %s", filepath)

    # ----------------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _build_da(
        time: npt.NDArray,
        mz: dict[int, npt.NDArray],
        values: dict[int, npt.NDArray],
        block_meta: Optional[dict[int, dict[str, Any]]],
        timestamps: Optional[pd.DatetimeIndex],
        tos_start: Optional[str | pd.Timestamp],
        cycle: Optional[npt.NDArray],
    ) -> xr.Dataset:
        
        time_coords: dict[str, Any] = {"time": time}
        if timestamps is not None:
            time_coords["timestamp"] = ("time", np.asarray(timestamps))
        if cycle is not None:
            time_coords["cycle"] = ("time", np.asarray(cycle, dtype=int))

        data_vars: dict[str, xr.DataArray] = {}
        for block_id in sorted(mz.keys()):
            mz_arr = np.asarray(mz[block_id], dtype=float)
            val_arr = np.asarray(values[block_id], dtype=float)
            unit = (block_meta or {}).get(block_id, {}).get("unit", "")
            data_vars[f"block_{block_id}"] = xr.DataArray(
                data=val_arr,
                coords={**time_coords, "mz": mz_arr},
                dims=["time", "mz"],
                attrs={"unit": unit, "block_id": block_id, "tos_start": tos_start},
            )

        return xr.Dataset(data_vars)

    # ----------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        time: npt.NDArray,
        mz: dict[int, npt.NDArray],
        values: dict[int, npt.NDArray],
        block_meta: Optional[dict[int, dict[str, Any]]] = None,
        timestamps: Optional[pd.DatetimeIndex] = None,
        cycle: Optional[npt.NDArray] = None,
        tos_start: Optional[str | pd.Timestamp] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "MSData":
        time = np.asarray(time, dtype=float)
        if time.ndim != 1:
            raise ValueError("time must be 1-D")
        if mz.keys() != values.keys():
            raise ValueError("mz and values must have the same block IDs")
        for block_id in mz:
            mz_arr = np.asarray(mz[block_id], dtype=float)
            val_arr = np.asarray(values[block_id], dtype=float)
            if val_arr.shape != (time.size, mz_arr.size):
                raise ValueError(
                    f"Block {block_id}: values shape {val_arr.shape} != "
                    f"(n_time={time.size}, n_mz={mz_arr.size})"
                )
        if tos_start is not None:
            if isinstance(tos_start, str):
                tos_start = pd.to_datetime(tos_start)

        ds = cls._build_da(time, mz, values, block_meta, timestamps, tos_start, cycle)
        return cls(ds=ds, block_meta=block_meta or {}, metadata=metadata or {})

    @classmethod
    def from_quadstar_asc(
        cls,
        filepath: Union[str, Path],
        tos_start: Optional[str | pd.Timestamp] = None,
        drop_threshold_cols: bool = True,
        tz_str: str = "Europe/Amsterdam",
    ) -> "MSData":
        meta, df = quadstar.read_export(filepath, drop_threshold_cols=drop_threshold_cols, tz_str=tz_str)
    
        # ---- time axis -----------------------------------------------
        if "RelTime[s]" in df.columns:
            time = df["RelTime[s]"].to_numpy(dtype=float)
        elif "Cycle" in df.columns:
            logger.warning("RelTime[s] not found — using Cycle index as time axis")
            time = df["Cycle"].to_numpy(dtype=float)
        else:
            logger.warning("Neither RelTime[s] nor Cycle found — using integer index")
            time = np.arange(len(df), dtype=float)

        timestamps: Optional[pd.DatetimeIndex] = None
        if "Timestamp" in df.columns:
            timestamps = pd.DatetimeIndex(df["Timestamp"])

        cycle_arr: Optional[npt.NDArray] = None
        if "Cycle" in df.columns:
            cycle_arr = df["Cycle"].to_numpy(dtype=int)

        # ---- per-block m/z and values --------------------------------
        column_map = meta.get("column_map", {})
        source_by_new = dict(zip(column_map["new"], column_map["source"]))

        mz_dict: dict[int, list[float]] = {}
        val_dict: dict[int, list[npt.NDArray]] = {}

        for col in df.columns:
            source = source_by_new.get(col, "meta")
            if not source.isdigit():
                continue
            block_id = int(source)
            if not col.startswith("m"):
                continue
            try:
                mass = float(col[1:])
            except ValueError:
                continue
            mz_dict.setdefault(block_id, []).append(mass)
            val_dict.setdefault(block_id, []).append(df[col].to_numpy(dtype=float))

        if not mz_dict:
            raise ValueError(
                "No data columns could be assigned to a datablock. "
                "Check that column_map was built correctly by the parser."
            )

        # Sort each block by m/z before handing to _build_da
        mz_arrays: dict[int, npt.NDArray] = {}
        val_arrays: dict[int, npt.NDArray] = {}
        for block_id in mz_dict:
            mz_vals = np.asarray(mz_dict[block_id], dtype=float)
            val_cols = np.column_stack(val_dict[block_id])
            sort_idx = np.argsort(mz_vals)
            mz_arrays[block_id] = mz_vals[sort_idx]
            val_arrays[block_id] = val_cols[:, sort_idx]

        ds = cls._build_da(
            time=time,
            mz=mz_arrays,
            values=val_arrays,
            block_meta=meta.get("datablocks", {}),
            timestamps=timestamps,
            tos_start=tos_start,
            cycle=cycle_arr,
        )
        return cls(
            ds=ds,
            block_meta=meta.get("datablocks", {}),
            metadata={k: v for k, v in meta.items() if k != "datablocks"},
        )

    @classmethod
    def from_netcdf(cls, filepath: Union[str, Path]) -> "MSData":
        """Reload an MSData previously saved with to_netcdf()."""
        ds = xr.open_dataset(filepath)
        return cls(ds=ds, metadata=dict(ds.attrs))

    # ----------------------------------------------------------------
    # Dunder helpers
    # ----------------------------------------------------------------

    def __repr__(self) -> str:
        t = self.time
        t_range = f"{t.min():.1f}–{t.max():.1f} s" if t.size else "empty"
        blocks_summary = ", ".join(
            f"block_{bid} ({self.block_type(bid)}, {self.unit(bid)}, "
            f"{self.mz(bid).size} channels)"
            for bid in self.block_ids
        )
        return (
            f"MSData("
            f"n_time={self.n_time}, "
            f"time={t_range}, "
            f"blocks=[{blocks_summary}]"
            f")"
        )

    def __len__(self) -> int:
        return self.n_time