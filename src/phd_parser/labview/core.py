import logging
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union, List

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, field_validator

from phd_parser._selection import EmptyWindow, window_mask
from phd_parser._time import TimeOriginMixin
from phd_parser.history import HistoryMixin
from phd_parser.history import HISTORY_KEY, decode_attrs, encode_attrs, get_history, records_step
from phd_parser.labview.b67box5 import read as read_b67box5
from phd_parser.naming.species.convention import ConventionLike, Unknown, get_convention

logger = logging.getLogger(__name__)




class LVData(TimeOriginMixin, HistoryMixin, BaseModel):
    """Pydantic wrapper for LabView process-data exported as a tab-separated log.

    The backing store is an ``xr.Dataset`` with a single ``tos`` dimension
    (elapsed seconds since ``tos_start``).  Each recorded channel is a data
    variable whose ``.attrs`` carry per-channel metadata (unit, group, species,
    location, etc.).  Absolute timestamps are derived on demand from ``tos``
    plus ``tos_start``; they are not stored as a coordinate.

    All processing methods are immutable: they return a new ``LVData`` instance
    and leave the original unchanged.
    """

    def _new(self, ds: xr.Dataset) -> "LVData":
        """A new instance around ``ds``, keeping the metadata."""
        return LVData(ds=ds, metadata=self.metadata)

    def _tos_start_raw(self) -> Any:
        """The origin lives in ``metadata`` (mirrored into ``ds.attrs``)."""
        return self.metadata.get("tos_start")

    def _with_tos_start(self, ds: xr.Dataset, tos_start: Optional[pd.Timestamp]) -> "LVData":
        """Stamp (or drop) ``tos_start`` on both the metadata dict and the dataset attrs."""
        metadata = dict(self.metadata)
        attrs = dict(ds.attrs)
        for store in (metadata, attrs):
            if tos_start is None:
                store.pop("tos_start", None)
            else:
                store["tos_start"] = tos_start.isoformat()
        ds.attrs = attrs
        return LVData(ds=ds, metadata=metadata)


    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        ignored_types=(cached_property,),
    )

    # ----------------------------------------------------------------
    # Fields
    # ----------------------------------------------------------------

    # Core data: dims: ('tos',). Each recorded channel is a data_var.
    # 'tos' is elapsed seconds since tos_start (mirrors IRData convention).
    # Per-channel metadata (unit, group, species, ...) lives in each variable's attrs.
    # 'timestamps' is NOT stored: derived on demand from tos + tos_start.
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
        """Check the layout of ``ds``: a single ``tos`` dimension shared by every channel."""
        if not isinstance(v, xr.Dataset):
            raise TypeError(f"'ds' must be an xr.Dataset, got {type(v)}")
        if "tos" not in v.dims:
            raise ValueError("Dataset must have a 'tos' dimension")
        return v

    # ----------------------------------------------------------------
    # Core properties
    # ----------------------------------------------------------------

    @property
    def history(self) -> list[dict[str, Any]]:
        """Every processing step that produced this object, oldest first (see :mod:`phd_parser.history`).

        Kept in ``metadata["history"]`` (mirrored into ``ds.attrs``) and saved by ``to_netcdf``.

        Returns
        -------
        list of dict
            ``{"step", "parameters", "time", "version"}`` per step.
        """
        return get_history(self.metadata)

    def _with_history(self, history: list[dict[str, Any]]) -> "LVData":
        """Copy carrying ``history`` in both ``metadata`` and ``ds.attrs`` (used by phd_parser.history)."""
        ds = self.ds.copy()
        ds.attrs = {**ds.attrs, HISTORY_KEY: history}
        return LVData(ds=ds, metadata={**self.metadata, HISTORY_KEY: history})

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
    # Immutable: selection
    # ----------------------------------------------------------------

    @records_step
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

    @records_step
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

    @records_step
    def select_tos_range(
        self,
        min_s: Optional[float] = None,
        max_s: Optional[float] = None,
        empty: EmptyWindow = "warn",
    ) -> "LVData":
        """Return a new ``LVData`` sliced to a time-on-stream window.

        Parameters
        ----------
        min_s : float, optional
            Lower bound of ``tos`` in seconds (inclusive).
        max_s : float, optional
            Upper bound of ``tos`` in seconds (inclusive).
        empty : {"warn", "raise"}, optional
            What to do when nothing falls inside the window: log it and
            return the empty selection (default), or raise ``ValueError``.

        Returns
        -------
        LVData
            New instance containing only samples within ``[min_s, max_s]``.

        Raises
        ------
        ValueError
            If the window is empty and ``empty="raise"``.
        """
        mask = window_mask(self.ds.coords["tos"].values, min_s, max_s,
                           what="sample", axis="tos", unit="s", empty=empty)
        return LVData(ds=self.ds.isel(tos=mask), metadata=self.metadata)

    # ----------------------------------------------------------------
    # Immutable: resampling / averaging
    # ----------------------------------------------------------------

    @records_step
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
            ``"mean"``).  NaN samples are skipped; ``first`` and ``last`` are
            the first and last recorded (non-NaN) sample of the bin.

        Returns
        -------
        LVData
            New instance on a uniform ``tos`` grid with bin-centre coordinates;
            a bin without samples is NaN.

        Raises
        ------
        ValueError
            If *step_s* is not positive or *method* is unknown.
        """
        if step_s <= 0:
            raise ValueError("step_s must be > 0")

        tos = self.tos
        bin_idx = np.floor((tos - tos[0]) / step_s).astype(int)
        n_bins = int(bin_idx.max()) + 1
        new_tos = tos[0] + (np.arange(n_bins) + 0.5) * step_s

        if method not in ("mean", "median", "first", "last"):
            raise ValueError(f"method must be 'mean', 'median', 'first' or 'last', got {method!r}")

        def _bin(values: npt.NDArray) -> npt.NDArray:
            # one grouped pass per channel; NaN samples (a channel a file did not record) are skipped
            grouped = pd.Series(values).groupby(bin_idx).agg(method)
            return grouped.reindex(range(n_bins)).to_numpy(dtype=float)

        return self._apply_per_channel(
            _bin,
            new_tos=new_tos,
            extra_metadata={"resample_step_s": step_s, "resample_method": method},
        )

    @records_step
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
        return self._apply_per_channel(
            # centred window that shrinks at the edges and skips NaN samples, so the
            # ends are not pulled towards zero and a NaN gap does not spread
            lambda v: pd.Series(v).rolling(window_size, center=True, min_periods=1).mean().to_numpy(dtype=float),
            extra_metadata={"smooth_moving_window": window_size},
        )

    # ----------------------------------------------------------------
    # Immutable: naming
    # ----------------------------------------------------------------

    @records_step
    def rename_species(self, convention: ConventionLike = None, unknown: Unknown = "keep") -> "LVData":
        """Return a new ``LVData`` with each channel's ``species`` attribute renamed to a naming convention.

        Channel names and values are untouched; only the ``species`` entry of
        the per-channel metadata changes.

        Parameters
        ----------
        convention : None, str, Path, dict or SpeciesConvention, optional
            ``None`` (default) for the standard convention of
            :mod:`phd_parser.naming.species`, a JSON file, an already read
            JSON object, or a convention.
        unknown : {"keep", "raise", "none"}, optional
            What to do with species the convention does not know (default:
            keep the name and warn).

        Returns
        -------
        LVData
            New instance; ``metadata["species_convention"]`` names the convention.
        """
        resolved = get_convention(convention)
        ds = self.ds.copy()
        for name in ds.data_vars:
            species = ds[name].attrs.get("species")
            if species:
                ds[name].attrs = {**ds[name].attrs, "species": resolved.get_canonical(str(species), unknown)}
        metadata = {**self.metadata, "species_convention": resolved.name}
        ds.attrs = dict(metadata)
        return LVData(ds=ds, metadata=metadata)

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
        # tos_start in ds.attrs (via metadata) round-trips automatically.
        # NetCDF cannot store None: an attribute that is None is left out,
        # which reads back as absent (e.g. `filename_timestamp` when the
        # export was read from a folder rather than a single file).
        # The same holds for per-channel attributes (a channel without a unit).
        # metadata is written too: some keys (species_convention, ...) live only there.
        # Nested values (the history, lists of text) are JSON-encoded, see phd_parser.history.
        ds = self.ds.copy()
        ds.attrs = encode_attrs({k: v for k, v in {**ds.attrs, **self.metadata}.items() if v is not None})
        for name in ds.variables:
            ds[name].attrs = encode_attrs({k: v for k, v in ds[name].attrs.items() if v is not None})
        ds.to_netcdf(filepath, engine="h5netcdf")
        logger.debug("Saved NetCDF → %s", filepath)

    # ----------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------

    @classmethod
    @records_step
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
    @records_step
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
        # read into memory and close, so the file is not held open (Windows locks it)
        with xr.open_dataset(filepath) as handle:
            ds = handle.load()
        ds.attrs = decode_attrs(dict(ds.attrs))
        for name in ds.variables:
            ds[name].attrs = decode_attrs(dict(ds[name].attrs))
        return cls(ds=ds, metadata=dict(ds.attrs))

    @classmethod
    @records_step
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
