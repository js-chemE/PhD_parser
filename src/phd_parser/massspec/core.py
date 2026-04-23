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
 
# The primary block (ID 0) is always the m/z block and uses the canonical
# dim names. Auxiliary blocks (pressure, temperature, ...) get per-block
# channel dims.
PRIMARY_BLOCK_ID = 0
 
def _channel_dim(block_id: int) -> str:
    """Name of the channel dim for a given block."""
    return "mz" if block_id == PRIMARY_BLOCK_ID else f"ch_{block_id}"

class MSData(BaseModel):
    """Mass-spec data container.
 
    Everything lives on the xr.Dataset. There is no separate metadata dict.
 
    Layout
    ------
    ds.coords:
        'cycle' (int, dim coord)          — shared scan index across blocks
        'tos'   (float s, non-dim coord)  — time on stream along 'cycle' (optional)
    ds.data_vars (one per datablock, named 'block_{id}'):
        block_0 (m/z): dims = ('cycle', 'mz'),   mz coord in Da
        block_N (aux): dims = ('cycle', 'ch_N'), ch_N coord in block-native units
        Each DataArray's .attrs may hold:
            'unit'            — e.g. 'A', 'mbar', ...
            'type'            — e.g. 'MID', 'analog', ...
            'block_id'        — int, redundant with the name
            'channel_labels'  — np.ndarray[str] of original column labels
    ds.attrs:
        'tos_start'         — ISO-8601 string (optional)
        'trace_corrections' — list[dict], audit trail (optional)
        plus any file-level metadata forwarded by the parser
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        ignored_types=(cached_property,),
    )

    # ----------------------------------------------------------------
    # Fields
    # ----------------------------------------------------------------
 
    ds: xr.Dataset = Field(
        description="Dataset with shared 'cycle' dim and per-block channel dims."
    )
 
    # ----------------------------------------------------------------
    # Validators
    # ----------------------------------------------------------------
 
    @field_validator("ds", mode="before")
    @classmethod
    def _validate_dataset(cls, v: Any) -> xr.Dataset:
        if not isinstance(v, xr.Dataset):
            raise TypeError(f"'ds' must be an xr.Dataset, got {type(v)}")
        if not v.data_vars:
            raise ValueError("Dataset must contain at least one DataArray.")
        if "cycle" not in v.coords:
            raise ValueError("Dataset must have a 'cycle' coord.")
        return v
 
    @model_validator(mode="after")
    def _validate_block_dims(self) -> "MSData":
        for name, da in self.ds.data_vars.items():
            try:
                bid = int(name.split("_", 1)[1])
            except (IndexError, ValueError) as e:
                raise ValueError(f"DataArray name '{name}' is not 'block_<int>'") from e
            expected = ("cycle", _channel_dim(bid))
            if da.dims != expected:
                raise ValueError(
                    f"DataArray '{name}' must have dims {expected}, got {da.dims}"
                )
        if f"block_{PRIMARY_BLOCK_ID}" not in self.ds.data_vars:
            raise ValueError(
                f"Dataset must contain the primary m/z block "
                f"'block_{PRIMARY_BLOCK_ID}'. Blocks found: {self.block_ids}"
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
    def cycle(self) -> npt.NDArray:
        return self.ds.coords["cycle"].values
 
    @property
    def n_cycle(self) -> int:
        return self.cycle.size
 
    def __len__(self) -> int:
        return self.n_cycle
 
    @property
    def tos(self) -> Optional[npt.NDArray]:
        """Time on stream [s] since tos_start. None if not available."""
        if "tos" in self.ds.coords:
            return self.ds.coords["tos"].values
        return None
 
    @property
    def tos_start(self) -> Optional[pd.Timestamp]:
        """Absolute timestamp corresponding to tos=0, if set."""
        iso = self.ds.attrs.get("tos_start")
        return pd.to_datetime(iso) if iso else None
 
    @property
    def timestamps(self) -> Optional[pd.DatetimeIndex]:
        """Absolute timestamps reconstructed from tos + tos_start."""
        tos = self.tos
        tos_start = self.tos_start
        if tos is None or tos_start is None:
            return None
        return pd.DatetimeIndex(tos_start + pd.to_timedelta(tos, unit="s"))

    # ----------------------------------------------------------------
    # Block / channel access
    # ----------------------------------------------------------------
 
    def _block(self, block_id: int) -> xr.DataArray:
        name = f"block_{block_id}"
        if name not in self.ds:
            raise KeyError(f"Block {block_id} not found. Available: {self.block_ids}")
        return self.ds[name]
 
    def channels(self, block_id: int = PRIMARY_BLOCK_ID) -> npt.NDArray:
        """Channel coord values for a block (m/z for block 0, arbitrary for others)."""
        da = self._block(block_id)
        return da.coords[_channel_dim(block_id)].values
 
    def mz(self) -> npt.NDArray:
        """m/z grid of the primary block."""
        return self.channels(PRIMARY_BLOCK_ID)
 
    def values(self, block_id: int = PRIMARY_BLOCK_ID) -> npt.NDArray:
        return self._block(block_id).values
 
    def unit(self, block_id: int = PRIMARY_BLOCK_ID) -> str:
        return str(self._block(block_id).attrs.get("unit", "?"))
 
    def block_type(self, block_id: int = PRIMARY_BLOCK_ID) -> str:
        return str(self._block(block_id).attrs.get("type", "?"))
 
    def channel_labels(self, block_id: int) -> Optional[list[str]]:
        """Original column labels for a block, if stored."""
        labels = self._block(block_id).attrs.get("channel_labels")
        return None if labels is None else [str(x) for x in np.atleast_1d(labels)]
 
    # ----------------------------------------------------------------
    # Cached derived quantities
    # ----------------------------------------------------------------
 
    @cached_property
    def _tic(self) -> npt.NDArray:
        return np.nansum(self._block(PRIMARY_BLOCK_ID).values, axis=1)
 
    def tic(self) -> npt.NDArray:
        """Total ion current vs cycle — only meaningful for the m/z block."""
        return self._tic

    # ----------------------------------------------------------------
    # Trace / spectrum extraction (primary m/z block)
    # ----------------------------------------------------------------
 
    @staticmethod
    def _check_mz_tolerance(
        targets: npt.NDArray,
        mz_grid: npt.NDArray,
        tolerance: Optional[float],
    ) -> None:
        if tolerance is None:
            return
        for t in np.atleast_1d(targets):
            dist = float(np.abs(mz_grid - t).min())
            if dist > tolerance:
                raise ValueError(
                    f"Requested m/z {t} is {dist:.3f} Da from the nearest grid "
                    f"point (tolerance: {tolerance})"
                )
 
    def get_trace(
        self,
        mz: float,
        method: Literal["nearest", "linear"] = "nearest",
        tolerance: Optional[float] = 0.2,
    ) -> xr.DataArray:
        """Intensity vs cycle for a single m/z from the primary block."""
        da = self._block(PRIMARY_BLOCK_ID)
        self._check_mz_tolerance(np.asarray([mz], dtype=float), da.coords["mz"].values, tolerance)
        return da.sel(mz=mz, method=method)
 
    def get_traces(
        self,
        mz_list: Sequence[float],
        method: Literal["nearest", "linear"] = "nearest",
        tolerance: Optional[float] = None,
    ) -> xr.DataArray:
        """Intensity vs cycle for multiple m/z values from the primary block."""
        targets = np.atleast_1d(np.asarray(mz_list, dtype=float))
        da = self._block(PRIMARY_BLOCK_ID)
        self._check_mz_tolerance(targets, da.coords["mz"].values, tolerance)
        return da.sel(mz=targets, method=method)
 
    def get_spectrum(self, cycle: int) -> xr.DataArray:
        """Full m/z spectrum at a single cycle (nearest), from the primary block."""
        return self._block(PRIMARY_BLOCK_ID).sel(cycle=cycle, method="nearest")
 
    def get_channel(
        self,
        block_id: int,
        channel: float,
        method: Literal["nearest", "linear"] = "nearest",
    ) -> xr.DataArray:
        """Single-channel trace from any block (e.g. pressure from block 1)."""
        da = self._block(block_id)
        dim = _channel_dim(block_id)
        return da.sel({dim: channel}, method=method)

    # ----------------------------------------------------------------
    # Immutable slicing
    # ----------------------------------------------------------------
 
    def select_tos_range(
        self,
        tos_start_seconds: Optional[float] = None,
        tos_end_seconds: Optional[float] = None,
    ) -> "MSData":
        """Return a new MSData restricted to [tos_start_seconds, tos_end_seconds] s."""
        tos = self.tos
        if tos is None:
            raise ValueError(
                "Cannot select by tos because 'tos' coord is not set "
                "(tos_start was not provided at construction)."
            )
        mask = np.ones(tos.size, dtype=bool)
        if tos_start_seconds is not None:
            mask &= tos >= tos_start_seconds
        if tos_end_seconds is not None:
            mask &= tos <= tos_end_seconds
        return MSData(ds=self.ds.isel(cycle=mask))
    
    # ----------------------------------------------------------------
    # Immutable baseline / trace correction
    # ----------------------------------------------------------------
 
    def correct_traces(
        self,
        mz: Union[None, Literal["all"], float, Sequence[float]] = "all",
        tolerance: Optional[float] = 0.2,
    ) -> "MSData":
        """Shift negative-valued traces of the m/z block up so their minimum is 0."""
        if mz is None:
            return MSData(ds=self.ds.copy(deep=True))
 
        new_ds = self.ds.copy(deep=True)
        name = f"block_{PRIMARY_BLOCK_ID}"
        arr = new_ds[name].values
        mz_grid = new_ds[name].coords["mz"].values
        shifts_log: dict[float, float] = {}
 
        if isinstance(mz, str):
            if mz != "all":
                raise ValueError(f"mz string argument must be 'all', got {mz!r}")
            mins = np.nanmin(arr, axis=0)
            shift = np.where(mins < 0, mins, 0.0)
            new_ds[name] = new_ds[name] - shift
            for i in np.where(shift < 0)[0]:
                shifts_log[float(mz_grid[i])] = float(-shift[i])
        else:
            try:
                targets = np.atleast_1d(np.asarray(mz, dtype=float))
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"mz must be None, 'all', a float, or a sequence of floats; got {mz!r}"
                ) from e
 
            self._check_mz_tolerance(targets, mz_grid, tolerance)
            for target in targets:
                idx = int(np.argmin(np.abs(mz_grid - target)))
                trace_min = np.nanmin(arr[:, idx])
                if trace_min < 0:
                    arr[:, idx] = arr[:, idx] - trace_min
                    shifts_log[float(mz_grid[idx])] = float(-trace_min)
            new_ds[name].values[...] = arr
 
        self._append_correction(new_ds, {"method": "min_shift", "shifts": shifts_log})
        return MSData(ds=new_ds)
 
    def baseline_subtract(
        self,
        tos_start_seconds: float,
        tos_end_seconds: float,
        block_id: Union[int, Literal["all"], None] = "all",
    ) -> "MSData":
        """Subtract per-channel mean over the given tos window. Immutable."""
        tos = self.tos
        if tos is None:
            raise ValueError(
                "Cannot baseline-subtract by tos range because 'tos' coord is not set."
            )
 
        mask = (tos >= tos_start_seconds) & (tos <= tos_end_seconds)
        if not mask.any():
            raise ValueError(
                f"Baseline window [{tos_start_seconds}, {tos_end_seconds}] s "
                f"contains no samples. tos range is "
                f"[{float(tos.min()):.1f}, {float(tos.max()):.1f}] s."
            )
 
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
            baseline = da.isel(cycle=mask).mean(dim="cycle", skipna=True)
            new_ds[name] = da - baseline
            chan_dim = _channel_dim(bid)
            baseline_log[bid] = {
                "mean_per_channel": {
                    float(c): float(b)
                    for c, b in zip(da.coords[chan_dim].values, baseline.values)
                },
                "n_samples": int(mask.sum()),
            }
 
        self._append_correction(
            new_ds,
            {
                "method": "baseline_subtract",
                "window_tos_s": [float(tos_start_seconds), float(tos_end_seconds)],
                "baselines": baseline_log,
            },
        )
        return MSData(ds=new_ds)
 
    @staticmethod
    def _append_correction(ds: xr.Dataset, entry: dict[str, Any]) -> None:
        history = list(ds.attrs.get("trace_corrections", []))
        history.append(entry)
        ds.attrs["trace_corrections"] = history

    # ----------------------------------------------------------------
    # Export
    # ----------------------------------------------------------------
 
    def to_csv(self, filepath: Union[str, Path], block_id: int = PRIMARY_BLOCK_ID) -> None:
        filepath = Path(filepath)
        da = self._block(block_id)
        chan_dim = _channel_dim(block_id)
        chan_label = "mz" if block_id == PRIMARY_BLOCK_ID else chan_dim
        df = pd.DataFrame(
            da.values,
            index=pd.Index(self.cycle, name="cycle"),
            columns=[f"{chan_label}_{c}" for c in da.coords[chan_dim].values],
        )
        if self.tos is not None:
            df.insert(0, "tos_s", self.tos)
        if self.timestamps is not None:
            df.insert(0, "timestamp", self.timestamps)
        df.to_csv(filepath)
        logger.debug("Saved block %d → %s", block_id, filepath)
 
    def to_netcdf(self, filepath: Union[str, Path]) -> None:
        """Save full dataset to NetCDF4. Reload with from_netcdf()."""
        self.ds.to_netcdf(filepath)
        logger.debug("Saved NetCDF → %s", filepath)

    # ----------------------------------------------------------------
    # Private: dataset builder
    # ----------------------------------------------------------------
 
    @staticmethod
    def _build_ds(
        cycle: npt.NDArray,
        channels: dict[int, npt.NDArray],
        values: dict[int, npt.NDArray],
        block_attrs: Optional[dict[int, dict[str, Any]]],
        tos: Optional[npt.NDArray],
        tos_start: Optional[pd.Timestamp],
        ds_attrs: Optional[dict[str, Any]] = None,
    ) -> xr.Dataset:
        cycle_coords: dict[str, Any] = {"cycle": np.asarray(cycle, dtype=int)}
        if tos is not None:
            cycle_coords["tos"] = ("cycle", np.asarray(tos, dtype=float))
 
        attrs: dict[str, Any] = dict(ds_attrs or {})
        if tos_start is not None:
            attrs["tos_start"] = pd.Timestamp(tos_start).isoformat()
 
        data_vars: dict[str, xr.DataArray] = {}
        for block_id in sorted(channels.keys()):
            chan_arr = np.asarray(channels[block_id], dtype=float)
            val_arr = np.asarray(values[block_id], dtype=float)
            chan_dim = _channel_dim(block_id)
 
            da_attrs: dict[str, Any] = {"block_id": int(block_id)}
            extra = (block_attrs or {}).get(block_id, {})
            for k, v in extra.items():
                if k == "channel_labels":
                    # Store as numpy string array for NetCDF compatibility.
                    da_attrs[k] = np.asarray(list(v), dtype="U")
                else:
                    da_attrs[k] = v
            # Ensure 'unit' always exists (even if empty) for predictable access.
            da_attrs.setdefault("unit", "")
 
            data_vars[f"block_{block_id}"] = xr.DataArray(
                data=val_arr,
                coords={**cycle_coords, chan_dim: chan_arr},
                dims=["cycle", chan_dim],
                attrs=da_attrs,
            )
 
        return xr.Dataset(data_vars, attrs=attrs)
 
    # ----------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------
 
    @classmethod
    def from_arrays(
        cls,
        cycle: npt.NDArray,
        channels: dict[int, npt.NDArray],
        values: dict[int, npt.NDArray],
        block_attrs: Optional[dict[int, dict[str, Any]]] = None,
        tos: Optional[npt.NDArray] = None,
        tos_start: Optional[Union[str, pd.Timestamp]] = None,
        ds_attrs: Optional[dict[str, Any]] = None,
    ) -> "MSData":
        cycle = np.asarray(cycle, dtype=int)
        if cycle.ndim != 1:
            raise ValueError("cycle must be 1-D")
        if channels.keys() != values.keys():
            raise ValueError("channels and values must have the same block IDs")
        if PRIMARY_BLOCK_ID not in channels:
            raise ValueError(
                f"The primary m/z block (id={PRIMARY_BLOCK_ID}) must be provided."
            )
        for block_id in channels:
            chan_arr = np.asarray(channels[block_id], dtype=float)
            val_arr = np.asarray(values[block_id], dtype=float)
            if val_arr.shape != (cycle.size, chan_arr.size):
                raise ValueError(
                    f"Block {block_id}: values shape {val_arr.shape} != "
                    f"(n_cycle={cycle.size}, n_channels={chan_arr.size})"
                )
        if tos is not None:
            tos = np.asarray(tos, dtype=float)
            if tos.shape != cycle.shape:
                raise ValueError(
                    f"tos shape {tos.shape} != cycle shape {cycle.shape}"
                )
        if isinstance(tos_start, str):
            tos_start = pd.to_datetime(tos_start)
 
        ds = cls._build_ds(cycle, channels, values, block_attrs, tos, tos_start, ds_attrs)
        return cls(ds=ds)
 
    @classmethod
    def from_quadstar_asc(
        cls,
        filepath: Union[str, Path],
        tos_start: Optional[Union[str, pd.Timestamp]] = None,
        drop_threshold_cols: bool = True,
        tz_str: str = "Europe/Amsterdam",
    ) -> "MSData":
        meta, df = quadstar.read_export(
            filepath, drop_threshold_cols=drop_threshold_cols, tz_str=tz_str
        )
 
        # ---- cycle axis ---------------------------------------------
        if "Cycle" in df.columns:
            cycle = df["Cycle"].to_numpy(dtype=int)
        else:
            logger.warning("Cycle column not found — using integer index")
            cycle = np.arange(len(df), dtype=int)
 
        # ---- tos (time on stream) -----------------------------------
        if isinstance(tos_start, str):
            tos_start = pd.to_datetime(tos_start)
 
        tos: Optional[npt.NDArray] = None
        if tos_start is not None and "Timestamp" in df.columns:
            timestamps = pd.DatetimeIndex(df["Timestamp"])
            tos = (timestamps - tos_start).total_seconds().to_numpy()
        elif "RelTime[s]" in df.columns:
            tos = df["RelTime[s]"].to_numpy(dtype=float)
            logger.info(
                "tos_start not provided — using RelTime[s] as tos with t=0 at file start."
            )
 
        # ---- per-block channel coords and values --------------------
        # Block 0 (m/z): columns start with "m" → parse as float m/z.
        # Other blocks: use positional index as channel coord; stash original
        # column labels in block attrs so they aren't lost.
        column_map = meta.get("column_map", {})
        source_by_new = dict(zip(column_map["new"], column_map["source"]))
 
        chan_dict: dict[int, list[float]] = {}
        val_dict: dict[int, list[npt.NDArray]] = {}
        label_dict: dict[int, list[str]] = {}
 
        for col in df.columns:
            source = source_by_new.get(col, "meta")
            if not source.isdigit():
                continue
            block_id = int(source)
 
            if block_id == PRIMARY_BLOCK_ID:
                if not col.startswith("m"):
                    continue
                try:
                    channel_val = float(col[1:])
                except ValueError:
                    continue
            else:
                channel_val = float(len(chan_dict.get(block_id, [])))
 
            chan_dict.setdefault(block_id, []).append(channel_val)
            val_dict.setdefault(block_id, []).append(df[col].to_numpy(dtype=float))
            label_dict.setdefault(block_id, []).append(col)
 
        if PRIMARY_BLOCK_ID not in chan_dict:
            raise ValueError(
                f"No columns could be assigned to the primary m/z block "
                f"(id={PRIMARY_BLOCK_ID}). Check the parser's column_map."
            )
 
        chan_arrays: dict[int, npt.NDArray] = {}
        val_arrays: dict[int, npt.NDArray] = {}
        for block_id in chan_dict:
            chan_vals = np.asarray(chan_dict[block_id], dtype=float)
            val_cols = np.column_stack(val_dict[block_id])
            if block_id == PRIMARY_BLOCK_ID:
                sort_idx = np.argsort(chan_vals)
                chan_arrays[block_id] = chan_vals[sort_idx]
                val_arrays[block_id] = val_cols[:, sort_idx]
            else:
                chan_arrays[block_id] = chan_vals
                val_arrays[block_id] = val_cols
 
        # Merge parser's per-block metadata with column-label stash.
        parser_blocks = dict(meta.get("datablocks", {}))
        block_attrs: dict[int, dict[str, Any]] = {}
        for bid in chan_arrays:
            attrs = dict(parser_blocks.get(bid, {}))
            if bid != PRIMARY_BLOCK_ID and bid in label_dict:
                attrs["channel_labels"] = list(label_dict[bid])
            block_attrs[bid] = attrs
 
        # File-level metadata → ds.attrs (drop 'datablocks' — that's now per-DA).
        ds_attrs = {k: v for k, v in meta.items() if k != "datablocks"}
 
        ds = cls._build_ds(
            cycle=cycle,
            channels=chan_arrays,
            values=val_arrays,
            block_attrs=block_attrs,
            tos=tos,
            tos_start=tos_start,
            ds_attrs=ds_attrs,
        )
        return cls(ds=ds)
 
    @classmethod
    def from_netcdf(cls, filepath: Union[str, Path]) -> "MSData":
        """Reload an MSData previously saved with to_netcdf()."""
        ds = xr.open_dataset(filepath)
        return cls(ds=ds)
 
    # ----------------------------------------------------------------
    # Dunder
    # ----------------------------------------------------------------
 
    def __repr__(self) -> str:
        tos = self.tos
        if tos is not None and tos.size:
            t_range = f"tos={tos.min():.1f}–{tos.max():.1f} s"
        elif self.n_cycle:
            t_range = f"cycle={self.cycle.min()}–{self.cycle.max()}"
        else:
            t_range = "empty"
        blocks_summary = ", ".join(
            f"block_{bid} ({self.block_type(bid)}, {self.unit(bid)}, "
            f"{self.channels(bid).size} ch)"
            for bid in self.block_ids
        )
        return (
            f"MSData(n_cycle={self.n_cycle}, {t_range}, "
            f"blocks=[{blocks_summary}])"
        )