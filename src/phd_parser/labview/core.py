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


def _to_timedelta(delta: Union[float, pd.Timedelta, str]) -> pd.Timedelta:
    """Read a plain number as seconds; hand anything else to pandas.Timedelta."""
    if isinstance(delta, (int, float, np.integer, np.floating)) and not isinstance(delta, bool):
        return pd.Timedelta(seconds=float(delta))
    return pd.Timedelta(delta)


class LVData(BaseModel):
    """Pydantic wrapper for LabView process-data exported as a tab-separated log.

    The backing store is an ``xr.Dataset`` with a single ``tos`` dimension
    (elapsed seconds since ``tos_start``).  Each recorded channel is a data
    variable whose ``.attrs`` carry per-channel metadata (unit, group, species,
    location, etc.).  Absolute timestamps are derived on demand from ``tos``
    plus ``tos_start``; they are not stored as a coordinate.

    All processing methods are immutable: they return a new ``LVData`` instance
    and leave the original unchanged.
    """

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
        """Names of all data variables (channels) in the dataset.

        Returns
        -------
        list of str
            Ordered list of channel names as they appear in the underlying
            ``xr.Dataset``.
        """
        return list(self.ds.data_vars)

    @property
    def n_samples(self) -> int:
        """Number of time samples along the ``tos`` dimension.

        Returns
        -------
        int
            Length of the ``tos`` coordinate.
        """
        return int(self.ds.sizes["tos"])

    @property
    def tos(self) -> npt.NDArray:
        """Elapsed seconds since ``tos_start`` for each sample.

        Returns
        -------
        numpy.ndarray
            1-D array of elapsed seconds (the ``tos`` coordinate values).
        """
        # Elapsed seconds since tos_start (the single source of truth in the Dataset)
        return self.ds.coords["tos"].values

    @property
    def tos_start(self) -> Optional[pd.Timestamp]:
        """Absolute start time of the run, parsed from ``metadata``.

        Returns
        -------
        pandas.Timestamp or None
            Start timestamp, or ``None`` if ``"tos_start"`` is absent from
            ``metadata``.
        """
        # Parsed on demand from metadata ISO string — survives all transformations
        raw = self.metadata.get("tos_start")
        if raw is None:
            return None
        return pd.Timestamp(raw)

    @property
    def timestamps(self) -> Optional[pd.DatetimeIndex]:
        """Absolute datetime for every sample, derived from ``tos`` and ``tos_start``.

        Returns
        -------
        pandas.DatetimeIndex or None
            Index of absolute timestamps, or ``None`` when ``tos_start`` is
            unavailable.
        """
        # Derived on demand from tos + tos_start; not stored as a coordinate
        if self.tos_start is None:
            return None
        return self.tos_start + pd.to_timedelta(self.tos, unit="s")

    @property
    def sampling_interval(self) -> Optional[float]:
        """Median time step between consecutive samples, in seconds.

        Returns
        -------
        float or None
            Median of ``numpy.diff(tos)``, or ``None`` when fewer than two
            samples are present.
        """
        # Median spacing in seconds; None if only one sample
        if self.n_samples < 2:
            return None
        return float(np.median(np.diff(self.tos)))

    # ----------------------------------------------------------------
    # Channel access
    # ----------------------------------------------------------------

    def get_channel(self, name: str) -> npt.NDArray:
        """Return the raw data array for a single channel.

        Parameters
        ----------
        name : str
            Channel name as it appears in ``self.channels``.

        Returns
        -------
        numpy.ndarray
            1-D array of channel values along the ``tos`` dimension.

        Raises
        ------
        KeyError
            If ``name`` is not present in the dataset.
        """
        if name not in self.ds.data_vars:
            raise KeyError(f"Channel {name!r} not found. Available: {self.channels}")
        return self.ds[name].values

    def get_channel_unit(self, name: str) -> Optional[str]:
        """Return the physical unit string stored in a channel's attributes.

        Parameters
        ----------
        name : str
            Channel name as it appears in ``self.channels``.

        Returns
        -------
        str or None
            Value of the ``"unit"`` attribute, or ``None`` if the attribute is
            absent.

        Raises
        ------
        KeyError
            If ``name`` is not present in the dataset.
        """
        if name not in self.ds.data_vars:
            raise KeyError(f"Channel {name!r} not found. Available: {self.channels}")
        return self.ds[name].attrs.get("unit")

    def filter_by_group(self, group: str) -> list[str]:
        """Return channel names whose ``"group"`` attribute matches *group*.

        Parameters
        ----------
        group : str
            Group label to filter on (e.g. ``"temperature"``, ``"flow"``).

        Returns
        -------
        list of str
            Channel names (possibly empty) whose ``attrs["group"] == group``.
        """
        return [
            name for name, da in self.ds.data_vars.items()
            if da.attrs.get("group") == group
        ]

    # ----------------------------------------------------------------
    # Immutable — time origin (tos_start)
    # ----------------------------------------------------------------

    def with_tos_start(self, tos_start: Union[pd.Timestamp, str]) -> "LVData":
        """Re-anchor ``tos`` to a new origin, leaving every absolute timestamp unchanged.

        Moves the zero of the ``tos`` axis: each value is shifted by minus the
        distance the origin moved, so ``tos_start + tos`` still resolves to the
        moment each sample was actually recorded.  This is what you want when
        the experiment's reference point changes (e.g. aligning to when gas
        flow started rather than when logging did).

        Parameters
        ----------
        tos_start : pandas.Timestamp or str
            New absolute origin.  Strings are parsed by ``pandas.Timestamp``.

        Returns
        -------
        LVData
            New instance with the new origin and rebased ``tos`` values.  If no
            origin was set before, ``tos`` is left untouched and simply
            anchored to the new one.

        See Also
        --------
        set_tos_start : Replace the origin *without* touching ``tos``.
        move_tos_start_by : Move the origin by a relative amount.
        """
        new_tos_start = pd.Timestamp(tos_start)
        old_tos_start = self.tos_start

        ds = self.ds.copy()
        if old_tos_start is None:
            logger.info(
                f"No previous tos_start: keeping tos as-is and anchoring it to {new_tos_start}."
            )
        else:
            shift_seconds = (new_tos_start - old_tos_start).total_seconds()
            ds = ds.assign_coords(tos=ds.coords["tos"] - shift_seconds)
        return self._with_tos_start_metadata(ds, new_tos_start)

    def set_tos_start(self, tos_start: Union[pd.Timestamp, str]) -> "LVData":
        """Replace the origin without touching ``tos`` — every absolute timestamp moves.

        Use this to correct a wrong origin: the elapsed times are right, the
        wall-clock they were anchored to was not.

        Parameters
        ----------
        tos_start : pandas.Timestamp or str
            New absolute origin.  Strings are parsed by ``pandas.Timestamp``.

        Returns
        -------
        LVData
            New instance with the new origin and unchanged ``tos`` values.

        See Also
        --------
        with_tos_start : Re-anchor ``tos`` so the absolute timestamps survive.
        """
        return self._with_tos_start_metadata(self.ds.copy(), pd.Timestamp(tos_start))

    def del_tos_start(self) -> "LVData":
        """Drop the origin, keeping ``tos`` as a purely relative axis.

        Returns
        -------
        LVData
            New instance without a ``tos_start``; ``timestamps`` becomes
            ``None``.  Returns an equivalent instance if none was set.
        """
        metadata = dict(self.metadata)
        metadata.pop("tos_start", None)
        ds = self.ds.copy()
        ds.attrs = dict(metadata)
        return LVData(ds=ds, metadata=metadata)

    def move_tos_start_by(self, delta: Union[float, pd.Timedelta, str]) -> "LVData":
        """Move the origin by a relative amount and re-anchor ``tos`` to it.

        Equivalent to ``with_tos_start(tos_start + delta)``: absolute
        timestamps never change, so moving the origin *later* by ``delta``
        makes every ``tos`` value *smaller* by ``delta``.  Pass a negative
        ``delta`` to move the origin earlier and grow the ``tos`` values.

        Parameters
        ----------
        delta : float or pandas.Timedelta or str
            How far to move the origin.  A plain number is read as seconds;
            anything else is passed to ``pandas.Timedelta`` (e.g. ``"90s"``,
            ``"1h30min"``).

        Returns
        -------
        LVData
            New instance with the moved origin and rebased ``tos`` values.

        Raises
        ------
        ValueError
            If no ``tos_start`` is set, so there is nothing to move.
        """
        if self.tos_start is None:
            raise ValueError(
                "No tos_start to move. Anchor the data first with set_tos_start(...)."
            )
        return self.with_tos_start(self.tos_start + _to_timedelta(delta))

    def _with_tos_start_metadata(self, ds: xr.Dataset, tos_start: pd.Timestamp) -> "LVData":
        """Stamp tos_start onto both the metadata dict and the dataset attrs."""
        metadata = {**self.metadata, "tos_start": tos_start.isoformat()}
        ds.attrs = {**ds.attrs, "tos_start": tos_start.isoformat()}
        return LVData(ds=ds, metadata=metadata)

    # ----------------------------------------------------------------
    # Immutable — selection
    # ----------------------------------------------------------------

    def select_channels(self, channels: List[str]) -> "LVData":
        """Return a new ``LVData`` containing only the specified channels.

        Parameters
        ----------
        channels : list of str
            Channel names to keep.

        Returns
        -------
        LVData
            New instance whose dataset is restricted to *channels*.

        Raises
        ------
        KeyError
            If any name in *channels* is not present in the dataset.
        """
        missing = [c for c in channels if c not in self.ds.data_vars]
        if missing:
            raise KeyError(f"Channel(s) not found: {missing}. Available: {self.channels}")
        return LVData(ds=self.ds[channels], metadata=self.metadata)

    def select_group(self, group: str) -> "LVData":
        """Return a new ``LVData`` containing only channels belonging to *group*.

        Parameters
        ----------
        group : str
            Group label to select (e.g. ``"flow"``, ``"pressure"``).

        Returns
        -------
        LVData
            New instance restricted to channels in the given group.

        Raises
        ------
        ValueError
            If no channels belong to *group*.
        """
        names = self.filter_by_group(group)
        if not names:
            raise ValueError(f"No channels in group {group!r}")
        return self.select_channels(names)

    def select_tos_range(
        self,
        min_s: Optional[float] = None,
        max_s: Optional[float] = None,
    ) -> "LVData":
        """Return a new ``LVData`` sliced to a time-on-stream window.

        Parameters
        ----------
        min_s : float, optional
            Lower bound of ``tos`` in seconds (inclusive).  If the value
            exceeds the data range, the first sample is used and a warning
            is emitted.
        max_s : float, optional
            Upper bound of ``tos`` in seconds (inclusive).  If the value is
            below the data range, the last sample is used and a warning is
            emitted.

        Returns
        -------
        LVData
            New instance containing only samples within ``[min_s, max_s]``.
        """
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
        """Bin samples into equal-width time intervals and aggregate.

        Parameters
        ----------
        step_s : float
            Bin width in seconds.
        method : {"mean", "median", "first", "last"}, optional
            Aggregation function applied within each bin (default is
            ``"mean"``).

        Returns
        -------
        LVData
            New instance on a uniform ``tos`` grid with bin-centre coordinates.

        Raises
        ------
        ValueError
            If *step_s* is not positive.
        """
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
        """Apply a uniform moving-average filter to every channel.

        Parameters
        ----------
        window_size : int, optional
            Number of samples in the rectangular kernel (default is ``5``).

        Returns
        -------
        LVData
            New instance with smoothed channel values; ``tos`` is unchanged.

        Raises
        ------
        ValueError
            If *window_size* is less than 1.
        """
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
        """Convert the dataset to a ``pandas.DataFrame`` indexed by ``tos``.

        Parameters
        ----------
        with_timestamps : bool, optional
            If ``True`` and ``tos_start`` is available, a ``"timestamp"``
            column is prepended with absolute datetimes (default is ``False``).

        Returns
        -------
        pandas.DataFrame
            DataFrame with one column per channel, indexed by elapsed seconds.
        """
        df = self.ds.to_dataframe()
        if with_timestamps and self.timestamps is not None:
            df.insert(0, "timestamp", self.timestamps)
        return df

    def to_csv(self, filepath: Union[str, Path]) -> None:
        """Write the dataset to a CSV file.

        If ``tos_start`` is available, an absolute ``"timestamp"`` column is
        included automatically.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Destination file path.
        """
        filepath = Path(filepath)
        self.to_dataframe(with_timestamps=self.tos_start is not None).to_csv(filepath)
        logger.debug("Saved CSV → %s", filepath)

    def to_netcdf(self, filepath: Union[str, Path]) -> None:
        """Persist the dataset to a NetCDF file.

        ``tos_start`` is stored in ``ds.attrs`` (via ``metadata``) and
        round-trips automatically on reload.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Destination file path.
        """
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
        """Construct an ``LVData`` from a ``pandas.DataFrame``.

        The ``tos`` coordinate is computed as elapsed seconds from
        *tos_start* (or the first timestamp when omitted).  Timezone
        awareness is reconciled automatically between the timestamp column
        and *tos_start*.

        Parameters
        ----------
        df : pandas.DataFrame
            Source dataframe.  Must contain a datetime-parseable column
            named *timestamp_col*.
        timestamp_col : str, optional
            Name of the column holding absolute timestamps (default is
            ``"timestamp"``).
        tos_start : pandas.Timestamp, optional
            Reference time for computing ``tos``.  Defaults to the first
            timestamp in the column.
        channel_meta : dict, optional
            Mapping of channel name to attribute dict (e.g.
            ``{"T": {"unit": "K", "group": "temperature"}}``).
        metadata : dict, optional
            Additional provenance key/value pairs stored in ``self.metadata``
            and ``ds.attrs``.

        Returns
        -------
        LVData
            New instance built from *df*.

        Raises
        ------
        ValueError
            If *timestamp_col* is not a column of *df*.
        """
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
        """Load an ``LVData`` from a NetCDF file previously saved with ``to_netcdf``.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to the NetCDF file.

        Returns
        -------
        LVData
            Reconstructed instance.
        """
        ds = xr.open_dataset(filepath)
        return cls(ds=ds, metadata=dict(ds.attrs))

    @classmethod
    def from_b67_box5_txt(
        cls,
        filepath: Union[str, Path],
        tos_start: Optional[pd.Timestamp] = None,
        keep_unknown_channels: bool = False,
    ) -> "LVData":
        """Parse a LabView export from building 67, box 5 (high-pressure setup).

        Handles every vintage of the export: the recorded channel set has
        changed over time (``"F1 CO PV"`` was added in 2026-07), and a
        directory may hold files from either side of such a change.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to a single tab-separated ``.txt`` file or a directory
            containing multiple such files to be concatenated.
        tos_start : pandas.Timestamp, optional
            Reference time for computing ``tos``.  Defaults to the timestamp
            of the first data row.
        keep_unknown_channels : bool, optional
            Read columns the parser does not recognise in with empty metadata
            instead of skipping them (default is ``False``).

        Returns
        -------
        LVData
            New instance populated with the recognised channels found in the
            file(s).  Channels a file did not record are NaN over its rows.
        """
        df, channel_meta, file_meta = read_b67box5(
            filepath,
            tos_start=None,
            sep="\t",
            header=0,
            tzinfo="Europe/Amsterdam",
            keep_unknown_channels=keep_unknown_channels,
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
