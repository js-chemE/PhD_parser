import logging
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union, List

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, field_validator

from phd_parser.labview.b67box5 import read as read_b67box5

logger = logging.getLogger(__name__)


class LVData(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        ignored_types=(cached_property,),
    )

    # ----------------------------------------------------------------
    # Fields
    # ----------------------------------------------------------------

    # Core data — dims: ('tos',). Each recorded channel is a data_var.
    # 'tos' is elapsed seconds since tos_start (mirrors IRData convention).
    # Per-channel metadata (unit, group, species, ...) lives in each variable's attrs.
    # 'timestamps' is NOT stored — derived on demand from tos + tos_start.
    ds: xr.Dataset = Field(
        description=(
            "xarray Dataset with one dim 'tos' (seconds since tos_start). "
            "Each channel is a data variable with its own .attrs (unit, group, ...)."
        )
    )

    # 'tos_start' lives here as an ISO string so it survives all transformations.
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ----------------------------------------------------------------
    # Validators
    # ----------------------------------------------------------------

    @field_validator("ds", mode="before")
    @classmethod
    def validate_ds(cls, v: Any) -> xr.Dataset:
        if not isinstance(v, xr.Dataset):
            raise TypeError(f"'ds' must be an xr.Dataset, got {type(v)}")
        if "tos" not in v.dims:
            raise ValueError("Dataset must have a 'tos' dimension")
        if v.sizes["tos"] == 0:
            raise ValueError("Dataset 'tos' dimension is empty")
        return v

    # ----------------------------------------------------------------
    # Core properties
    # ----------------------------------------------------------------

    @property
    def channels(self) -> list[str]:
        return list(self.ds.data_vars)

    @property
    def n_samples(self) -> int:
        return int(self.ds.sizes["tos"])

    @property
    def tos(self) -> npt.NDArray:
        # Elapsed seconds since tos_start (the single source of truth in the Dataset)
        return self.ds.coords["tos"].values

    @property
    def tos_start(self) -> Optional[pd.Timestamp]:
        # Parsed on demand from metadata ISO string — survives all transformations
        raw = self.metadata.get("tos_start")
        if raw is None:
            return None
        return pd.Timestamp(raw)

    @property
    def timestamps(self) -> Optional[pd.DatetimeIndex]:
        # Derived on demand from tos + tos_start; not stored as a coordinate
        if self.tos_start is None:
            return None
        return self.tos_start + pd.to_timedelta(self.tos, unit="s")

    @property
    def sampling_interval(self) -> Optional[float]:
        # Median spacing in seconds; None if only one sample
        if self.n_samples < 2:
            return None
        return float(np.median(np.diff(self.tos)))

    # ----------------------------------------------------------------
    # Channel access
    # ----------------------------------------------------------------

    def get_channel(self, name: str) -> npt.NDArray:
        if name not in self.ds.data_vars:
            raise KeyError(f"Channel {name!r} not found. Available: {self.channels}")
        return self.ds[name].values

    def get_channel_unit(self, name: str) -> Optional[str]:
        if name not in self.ds.data_vars:
            raise KeyError(f"Channel {name!r} not found. Available: {self.channels}")
        return self.ds[name].attrs.get("unit")

    def filter_by_group(self, group: str) -> list[str]:
        return [
            name for name, da in self.ds.data_vars.items()
            if da.attrs.get("group") == group
        ]

    # ----------------------------------------------------------------
    # Immutable — selection
    # ----------------------------------------------------------------

    def select_channels(self, channels: List[str]) -> "LVData":
        missing = [c for c in channels if c not in self.ds.data_vars]
        if missing:
            raise KeyError(f"Channel(s) not found: {missing}. Available: {self.channels}")
        return LVData(ds=self.ds[channels], metadata=self.metadata)

    def select_group(self, group: str) -> "LVData":
        names = self.filter_by_group(group)
        if not names:
            raise ValueError(f"No channels in group {group!r}")
        return self.select_channels(names)

    def select_tos_range(
        self,
        min_s: Optional[float] = None,
        max_s: Optional[float] = None,
    ) -> "LVData":
        ds = self.ds
        tos = ds.coords["tos"].values
        if min_s is not None:
            if not np.any(tos >= min_s):
                min_s = tos[0]
                logger.warning(f"min_s exceeds data range; using min_s={min_s:.1f}s")
            ds = ds.isel(tos=tos >= min_s)
            tos = ds.coords["tos"].values
        if max_s is not None:
            if not np.any(tos <= max_s):
                max_s = tos[-1]
                logger.warning(f"max_s below data range; using max_s={max_s:.1f}s")
            ds = ds.isel(tos=tos <= max_s)
        return LVData(ds=ds, metadata=self.metadata)

    # ----------------------------------------------------------------
    # Immutable — resampling / averaging
    # ----------------------------------------------------------------

    def resample(
        self,
        step_s: float,
        method: Literal["mean", "median", "first", "last"] = "mean",
    ) -> "LVData":
        if step_s <= 0:
            raise ValueError("step_s must be > 0")

        tos = self.tos
        bin_idx = np.floor((tos - tos[0]) / step_s).astype(int)
        n_bins = int(bin_idx.max()) + 1
        new_tos = tos[0] + (np.arange(n_bins) + 0.5) * step_s

        agg: dict[str, Callable[[np.ndarray], float]] = {
            "mean": lambda g: g.mean() if g.size else np.nan,
            "median": lambda g: np.median(g) if g.size else np.nan,
            "first": lambda g: g[0] if g.size else np.nan,
            "last": lambda g: g[-1] if g.size else np.nan,
        }
        reduce = agg[method]

        def _bin(values: npt.NDArray) -> npt.NDArray:
            return np.array([reduce(values[bin_idx == i]) for i in range(n_bins)])

        return self._apply_per_channel(
            _bin,
            new_tos=new_tos,
            extra_metadata={"resample_step_s": step_s, "resample_method": method},
        )

    def smooth_moving(self, window_size: int = 5) -> "LVData":
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        kernel = np.ones(window_size) / window_size
        return self._apply_per_channel(
            lambda v: np.convolve(v, kernel, mode="same"),
            extra_metadata={"smooth_moving_window": window_size},
        )

    # ----------------------------------------------------------------
    # Export
    # ----------------------------------------------------------------

    def to_dataframe(self, with_timestamps: bool = False) -> pd.DataFrame:
        df = self.ds.to_dataframe()
        if with_timestamps and self.timestamps is not None:
            df.insert(0, "timestamp", self.timestamps)
        return df

    def to_csv(self, filepath: Union[str, Path]) -> None:
        filepath = Path(filepath)
        self.to_dataframe(with_timestamps=self.tos_start is not None).to_csv(filepath)
        logger.debug("Saved CSV → %s", filepath)

    def to_netcdf(self, filepath: Union[str, Path]) -> None:
        # tos_start in ds.attrs (via metadata) round-trips automatically
        self.ds.to_netcdf(filepath)
        logger.debug("Saved NetCDF → %s", filepath)

    # ----------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        tos_start: Optional[pd.Timestamp] = None,
        channel_meta: Optional[dict[str, dict[str, Any]]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "LVData":
        if timestamp_col not in df.columns:
            raise ValueError(f"Column {timestamp_col!r} not found in dataframe")

        ts = pd.to_datetime(df[timestamp_col])

        # ---- reconcile timezones between ts and tos_start ----
        # Subtracting tz-aware from tz-naive (or vice versa) raises.
        # Normalise both to the same awareness before computing tos.
        if tos_start is None:
            tos_start = ts.iloc[0]
        else:
            tos_start = pd.Timestamp(tos_start)

        ts_is_aware = ts.dt.tz is not None
        tos_is_aware = tos_start.tzinfo is not None
        if ts_is_aware and not tos_is_aware:
            logger.debug("Localising naive tos_start to %s", ts.dt.tz)
            tos_start = tos_start.tz_localize(ts.dt.tz)
        elif tos_is_aware and not ts_is_aware:
            logger.debug("Localising naive timestamp column to %s", tos_start.tzinfo)
            ts = ts.dt.tz_localize(tos_start.tzinfo)

        tos = (ts - tos_start).dt.total_seconds().to_numpy(dtype=float)

        # ---- build the Dataset in one shot ----
        # Passing `coords` at Dataset level (rather than per-DataArray) guarantees a
        # single shared 'tos' index and avoids MergeError from conflicting coords.
        # Skip reserved names ('tos', 'timestamp', 'timestamps') so parsers that
        # include them as columns don't collide with our coordinate.
        reserved = {timestamp_col, "tos", "timestamp", "timestamps"}
        channels = [c for c in df.columns if c not in reserved]
        channel_meta = channel_meta or {}

        data_vars = {
            ch: (
                ("tos",),
                df[ch].to_numpy(dtype=float),
                dict(channel_meta.get(ch, {})),
            )
            for ch in channels
        }

        meta = dict(metadata or {})
        meta["tos_start"] = tos_start.isoformat()

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={"tos": tos},
            attrs=meta,
        )
        return cls(ds=ds, metadata=meta)

    @classmethod
    def from_netcdf(cls, filepath: Union[str, Path]) -> "LVData":
        ds = xr.open_dataset(filepath)
        return cls(ds=ds, metadata=dict(ds.attrs))

    @classmethod
    def from_b67_box5_txt(
        cls,
        filepath: Union[str, Path],
        tos_start: Optional[pd.Timestamp] = None,
    ) -> "LVData":
        """Parse a LabView export from building 67, box 5 (high-pressure setup)."""
        df, channel_meta, file_meta = read_b67box5(
            filepath,
            tos_start=None,
            sep="\t",
            header=0,
            tzinfo="Europe/Amsterdam",
        )
        return cls.from_dataframe(
            df,
            timestamp_col="timestamp",
            tos_start=tos_start,
            channel_meta=channel_meta,
            metadata=file_meta,
        )

    # ----------------------------------------------------------------
    # Dunder helpers
    # ----------------------------------------------------------------

    def __repr__(self) -> str:
        dur = self.tos[-1] - self.tos[0] if self.n_samples > 1 else 0.0
        ts_info = f", tos_start={self.tos_start}" if self.tos_start is not None else ""
        return (
            f"LVData(n_samples={self.n_samples}, channels={len(self.channels)}, "
            f"duration={dur:.1f}s{ts_info})"
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, name: str) -> npt.NDArray:
        return self.get_channel(name)

    # ----------------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------------

    def _apply_per_channel(
        self,
        transform: Callable[[npt.NDArray], npt.NDArray],
        new_tos: Optional[npt.NDArray] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> "LVData":
        """Apply a 1-D transform to each channel, preserving per-channel attrs.

        If ``new_tos`` is given, the result uses that coordinate; otherwise the
        original 'tos' coord is kept (elementwise case).
        """
        tos_coord = new_tos if new_tos is not None else self.ds.coords["tos"].values

        data_vars = {
            name: (("tos",), transform(da.values), dict(da.attrs))
            for name, da in self.ds.data_vars.items()
        }
        new_metadata = {**self.metadata, **(extra_metadata or {})}
        new_ds = xr.Dataset(
            data_vars=data_vars,
            coords={"tos": tos_coord},
            attrs=new_metadata,
        )
        return LVData(ds=new_ds, metadata=new_metadata)