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

def _to_timedelta(delta: Union[float, pd.Timedelta, str]) -> pd.Timedelta:
    """Read a plain number as seconds; hand anything else to pandas.Timedelta."""
    if isinstance(delta, (int, float, np.integer, np.floating)) and not isinstance(delta, bool):
        return pd.Timedelta(seconds=float(delta))
    return pd.Timedelta(delta)

class MSData(BaseModel):
    """Mass-spectrometry data container backed by an ``xr.Dataset``.

    All data and metadata live on the ``ds`` field; there is no separate
    metadata dict.  Every processing method is immutable and returns a new
    ``MSData`` instance.

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
        """Sorted list of integer block IDs present in the dataset.

        Returns
        -------
        list[int]
            Block IDs in ascending order.
        """
        return sorted(int(n.split("_", 1)[1]) for n in self.ds.data_vars)

    @property
    def n_blocks(self) -> int:
        """Number of datablocks stored in the dataset.

        Returns
        -------
        int
            Count of datablocks.
        """
        return len(self.block_ids)

    @property
    def cycle(self) -> npt.NDArray:
        """Scan-index coordinate values shared across all blocks.

        Returns
        -------
        numpy.ndarray
            1-D integer array of cycle indices.
        """
        return self.ds.coords["cycle"].values

    @property
    def n_cycle(self) -> int:
        """Number of scan cycles in the dataset.

        Returns
        -------
        int
            Total cycle count.
        """
        return self.cycle.size

    def __len__(self) -> int:
        return self.n_cycle

    @property
    def tos(self) -> Optional[npt.NDArray]:
        """Time on stream in seconds since ``tos_start``.

        Returns
        -------
        numpy.ndarray or None
            1-D float array of elapsed seconds, or ``None`` if the ``tos``
            coordinate was not set at construction time.
        """
        if "tos" in self.ds.coords:
            return self.ds.coords["tos"].values
        return None

    @property
    def tos_start(self) -> Optional[pd.Timestamp]:
        """Absolute timestamp corresponding to ``tos = 0``.

        Returns
        -------
        pandas.Timestamp or None
            Start timestamp parsed from ``ds.attrs["tos_start"]``, or ``None``
            if the attribute is absent.
        """
        iso = self.ds.attrs.get("tos_start")
        return pd.to_datetime(iso) if iso else None

    @property
    def timestamps(self) -> Optional[pd.DatetimeIndex]:
        """Absolute timestamps reconstructed from ``tos`` and ``tos_start``.

        Returns
        -------
        pandas.DatetimeIndex or None
            Per-cycle wall-clock times, or ``None`` if either ``tos`` or
            ``tos_start`` is unavailable.
        """
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
        """Channel coordinate values for a given block.

        Parameters
        ----------
        block_id : int, optional
            Block to query.  Block 0 (default) returns m/z values in Da;
            auxiliary blocks return their native positional index.

        Returns
        -------
        numpy.ndarray
            1-D float array of channel coordinate values.

        Raises
        ------
        KeyError
            If ``block_id`` is not present in the dataset.
        """
        da = self._block(block_id)
        return da.coords[_channel_dim(block_id)].values

    def mz(self) -> npt.NDArray:
        """m/z grid of the primary block in Daltons.

        Returns
        -------
        numpy.ndarray
            1-D float array of m/z values.
        """
        return self.channels(PRIMARY_BLOCK_ID)

    def values(self, block_id: int = PRIMARY_BLOCK_ID) -> npt.NDArray:
        """Raw 2-D data array for a block with shape ``(n_cycle, n_channels)``.

        Parameters
        ----------
        block_id : int, optional
            Block to query (default is the primary m/z block 0).

        Returns
        -------
        numpy.ndarray
            2-D float array of shape ``(n_cycle, n_channels)``.

        Raises
        ------
        KeyError
            If ``block_id`` is not present in the dataset.
        """
        return self._block(block_id).values

    def unit(self, block_id: int = PRIMARY_BLOCK_ID) -> str:
        """Physical unit string for a block (e.g. ``"A"``, ``"mbar"``).

        Parameters
        ----------
        block_id : int, optional
            Block to query (default is the primary m/z block 0).

        Returns
        -------
        str
            Unit string from the DataArray's attributes, or ``"?"`` if unset.

        Raises
        ------
        KeyError
            If ``block_id`` is not present in the dataset.
        """
        return str(self._block(block_id).attrs.get("unit", "?"))

    def block_type(self, block_id: int = PRIMARY_BLOCK_ID) -> str:
        """Instrument block type string (e.g. ``"MID"``, ``"analog"``).

        Parameters
        ----------
        block_id : int, optional
            Block to query (default is the primary m/z block 0).

        Returns
        -------
        str
            Type string from the DataArray's attributes, or ``"?"`` if unset.

        Raises
        ------
        KeyError
            If ``block_id`` is not present in the dataset.
        """
        return str(self._block(block_id).attrs.get("type", "?"))

    def channel_labels(self, block_id: int) -> Optional[list[str]]:
        """Original column labels for a block as recorded in the source file.

        Parameters
        ----------
        block_id : int
            Block to query.

        Returns
        -------
        list[str] or None
            List of label strings, or ``None`` if no labels were stored.

        Raises
        ------
        KeyError
            If ``block_id`` is not present in the dataset.
        """
        labels = self._block(block_id).attrs.get("channel_labels")
        return None if labels is None else [str(x) for x in np.atleast_1d(labels)]

    # ----------------------------------------------------------------
    # Cached derived quantities
    # ----------------------------------------------------------------

    @cached_property
    def _tic(self) -> npt.NDArray:
        return np.nansum(self._block(PRIMARY_BLOCK_ID).values, axis=1)

    def tic(self) -> npt.NDArray:
        """Total ion current vs cycle, summed over all m/z channels.

        Only meaningful for the primary m/z block.  Result is cached after
        the first call.

        Returns
        -------
        numpy.ndarray
            1-D float array of length ``n_cycle``.
        """
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
        rolling_window: Optional[int] = None,
        normalize: Optional[Union[bool, tuple[float, float]]] = None,
    ) -> xr.DataArray:
        """Intensity vs cycle for a single m/z from the primary block.

        Parameters
        ----------
        mz : float
            Target m/z value in Daltons.
        method : {"nearest", "linear"}, optional
            xarray selection method (default is ``"nearest"``).
        tolerance : float or None, optional
            Maximum allowed distance in Da between ``mz`` and the nearest grid
            point.  Raises ``ValueError`` if exceeded.  Default is ``0.2``.
        rolling_window : int or None, optional
            Centered rolling mean over this many cycles, applied before
            normalization.  The underlying data is not modified.
        normalize : bool or tuple[float, float] or None, optional
            ``None`` / ``False`` — no normalization.
            ``True``             — scale to [0, 1] using the trace's own min/max.
            ``(vmin, vmax)``     — scale using the given fixed bounds.

        Returns
        -------
        xarray.DataArray
            1-D array indexed by ``cycle``.

        Raises
        ------
        ValueError
            If ``mz`` is farther than ``tolerance`` Da from the nearest grid
            point.
        """
        da = self._block(PRIMARY_BLOCK_ID)
        self._check_mz_tolerance(np.asarray([mz], dtype=float), da.coords["mz"].values, tolerance)
        result = da.sel(mz=mz, method=method)
        if normalize:
            if isinstance(normalize, tuple):
                vmin, vmax = float(normalize[0]), float(normalize[1])
            else:
                vmin = float(result.min())
                vmax = float(result.max())
            denom = vmax - vmin
            result = (result - vmin) / denom if denom != 0.0 else xr.zeros_like(result)

        if rolling_window is not None:
            result = result.rolling(cycle=rolling_window, center=True, min_periods=1).mean()

        return result

    def get_traces(
        self,
        mz_list: Sequence[float],
        method: Literal["nearest", "linear"] = "nearest",
        tolerance: Optional[float] = None,
        rolling_window: Optional[int] = None,
        normalize: Optional[Union[bool, tuple[float, float]]] = None,
    ) -> xr.DataArray:
        """Intensity vs cycle for multiple m/z values from the primary block.

        Delegates to :meth:`get_trace` per m/z so all parameters behave
        identically.  The tolerance check runs once up front.

        Parameters
        ----------
        mz_list : sequence of float
            Target m/z values in Daltons.
        method : {"nearest", "linear"}, optional
            xarray selection method (default is ``"nearest"``).
        tolerance : float or None, optional
            Maximum allowed distance in Da between each requested m/z and the
            nearest grid point.  ``None`` disables the check.
        rolling_window : int or None, optional
            Centered rolling mean applied per trace before normalization.
        normalize : bool or tuple[float, float] or None, optional
            ``True``         — each trace normalized independently to [0, 1].
            ``(vmin, vmax)`` — same fixed scale applied to every trace.

        Returns
        -------
        xarray.DataArray
            2-D array with dims ``(cycle, mz)``.

        Raises
        ------
        ValueError
            If any requested m/z exceeds ``tolerance`` from the nearest grid
            point.
        """
        targets = np.atleast_1d(np.asarray(mz_list, dtype=float))
        self._check_mz_tolerance(
            targets, self._block(PRIMARY_BLOCK_ID).coords["mz"].values, tolerance
        )
        traces = [
            self.get_trace(
                float(m),
                method=method,
                tolerance=None,
                rolling_window=rolling_window,
                normalize=normalize,
            )
            for m in targets
        ]
        return xr.concat(traces, dim="mz").transpose("cycle", "mz")

    def get_spectrum(self, cycle: int) -> xr.DataArray:
        """Full m/z spectrum at a single cycle from the primary block.

        Parameters
        ----------
        cycle : int
            Target cycle index; nearest-neighbour selection is used.

        Returns
        -------
        xarray.DataArray
            1-D array indexed by ``mz``.
        """
        return self._block(PRIMARY_BLOCK_ID).sel(cycle=cycle, method="nearest")

    def get_channel(
        self,
        block_id: int,
        channel: float,
        method: Literal["nearest", "linear"] = "nearest",
    ) -> xr.DataArray:
        """Single-channel time trace from any block.

        Parameters
        ----------
        block_id : int
            Block to query (e.g. ``1`` for an auxiliary pressure block).
        channel : float
            Channel coordinate value to select.
        method : {"nearest", "linear"}, optional
            xarray selection method (default is ``"nearest"``).

        Returns
        -------
        xarray.DataArray
            1-D array indexed by ``cycle``.

        Raises
        ------
        KeyError
            If ``block_id`` is not present in the dataset.
        """
        da = self._block(block_id)
        dim = _channel_dim(block_id)
        return da.sel({dim: channel}, method=method)

    # ----------------------------------------------------------------
    # Immutable — time origin (tos_start)
    # ----------------------------------------------------------------

    def with_tos_start(self, tos_start: Union[pd.Timestamp, str]) -> "MSData":
        """Re-anchor ``tos`` to a new origin, leaving every absolute timestamp unchanged.

        Moves the zero of the ``tos`` axis: each value is shifted by minus the
        distance the origin moved, so ``tos_start + tos`` still resolves to the
        moment each cycle was actually recorded.  This is what you want when
        the experiment's reference point changes (e.g. aligning to when gas
        flow started rather than when the MS did).

        Parameters
        ----------
        tos_start : pandas.Timestamp or str
            New absolute origin.  Strings are parsed by ``pandas.Timestamp``.

        Returns
        -------
        MSData
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
        elif "tos" in ds.coords:
            shift_seconds = (new_tos_start - old_tos_start).total_seconds()
            ds = ds.assign_coords(tos=ds.coords["tos"] - shift_seconds)
        ds.attrs = {**self.ds.attrs, "tos_start": new_tos_start.isoformat()}
        return MSData(ds=ds)

    def set_tos_start(self, tos_start: Union[pd.Timestamp, str]) -> "MSData":
        """Replace the origin without touching ``tos`` — every absolute timestamp moves.

        Use this to correct a wrong origin: the elapsed times are right, the
        wall-clock they were anchored to was not.

        Parameters
        ----------
        tos_start : pandas.Timestamp or str
            New absolute origin.  Strings are parsed by ``pandas.Timestamp``.

        Returns
        -------
        MSData
            New instance with the new origin and unchanged ``tos`` values.

        See Also
        --------
        with_tos_start : Re-anchor ``tos`` so the absolute timestamps survive.
        """
        ds = self.ds.copy()
        ds.attrs = {**self.ds.attrs, "tos_start": pd.Timestamp(tos_start).isoformat()}
        return MSData(ds=ds)

    def del_tos_start(self) -> "MSData":
        """Drop the origin, keeping ``tos`` as a purely relative axis.

        Returns
        -------
        MSData
            New instance without a ``tos_start``; ``timestamps`` becomes
            ``None``.  Returns an equivalent instance if none was set.
        """
        attrs = dict(self.ds.attrs)
        attrs.pop("tos_start", None)
        ds = self.ds.copy()
        ds.attrs = attrs
        return MSData(ds=ds)

    def move_tos_start_by(self, delta: Union[float, pd.Timedelta, str]) -> "MSData":
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
        MSData
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

    # ----------------------------------------------------------------
    # Immutable slicing
    # ----------------------------------------------------------------

    def select_tos_range(
        self,
        tos_start_seconds: Optional[float] = None,
        tos_end_seconds: Optional[float] = None,
    ) -> "MSData":
        """Return a new ``MSData`` restricted to a time-on-stream window.

        Parameters
        ----------
        tos_start_seconds : float or None, optional
            Lower bound in seconds (inclusive).  ``None`` means no lower bound.
        tos_end_seconds : float or None, optional
            Upper bound in seconds (inclusive).  ``None`` means no upper bound.

        Returns
        -------
        MSData
            New instance containing only cycles within the specified window.

        Raises
        ------
        ValueError
            If the ``tos`` coordinate is not set on the dataset.
        """
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
        """Shift negative-valued traces of the m/z block up so their minimum is 0.

        Parameters
        ----------
        mz : None, "all", float, or sequence of float, optional
            Which m/z channels to correct.  ``"all"`` (default) corrects every
            channel; ``None`` returns an unmodified copy; a scalar or sequence
            of floats corrects only those channels.
        tolerance : float or None, optional
            m/z grid-match tolerance in Da, applied when ``mz`` is a scalar or
            sequence.  Default is ``0.2``.

        Returns
        -------
        MSData
            New instance with corrections applied and the audit entry appended
            to ``ds.attrs["trace_corrections"]``.

        Raises
        ------
        ValueError
            If ``mz`` is the string ``"all"`` misspelled, or if a requested
            m/z exceeds ``tolerance`` from the nearest grid point.
        TypeError
            If ``mz`` is not a recognised type.
        """
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

    def mask_overloaded(
        self,
        threshold: float = 1e21,
        block_id: Union[int, Literal["all"], None] = "all",
    ) -> "MSData":
        """Replace detector-saturated values exceeding ``threshold`` with NaN.

        Values produced by an overloaded MS detector (e.g. ~1e38) are replaced
        with NaN so they do not distort downstream calculations.

        Parameters
        ----------
        threshold : float, optional
            Values strictly above this level are masked.  Default is ``1e21``.
        block_id : int, "all", or None, optional
            Which block(s) to process.  ``"all"`` or ``None`` (default ``"all"``)
            applies masking to every block.

        Returns
        -------
        MSData
            New instance with saturated values replaced by NaN and the audit
            entry appended to ``ds.attrs["trace_corrections"]``.

        Raises
        ------
        KeyError
            If a specific ``block_id`` integer is not present in the dataset.
        """
        if block_id is None or block_id == "all":
            target_blocks = self.block_ids
        else:
            if block_id not in self.block_ids:
                raise KeyError(f"Block {block_id} not found. Available: {self.block_ids}")
            target_blocks = [block_id]

        new_ds = self.ds.copy(deep=True)
        counts: dict[int, int] = {}

        for bid in target_blocks:
            name = f"block_{bid}"
            arr = new_ds[name].values
            mask = arr > threshold
            n = int(mask.sum())
            if n:
                arr[mask] = np.nan
                new_ds[name].values[...] = arr
            counts[bid] = n

        self._append_correction(
            new_ds,
            {"method": "mask_overloaded", "threshold": threshold, "n_masked": counts},
        )
        return MSData(ds=new_ds)

    def smooth_trace_rolling(
        self,
        window: int,
        block_id: Union[int, Literal["all"], None] = "all",
        center: bool = True,
        min_periods: Optional[int] = 1,
    ) -> "MSData":
        """Apply a centered rolling mean along the cycle dimension.

        Parameters
        ----------
        window : int
            Number of cycles to average over.  Must be at least 2.
        block_id : int, "all", or None, optional
            Which block(s) to smooth.  ``"all"`` or ``None`` (default ``"all"``)
            smooths every block.
        center : bool, optional
            If ``True`` (default) the window is centered on each cycle.
        min_periods : int or None, optional
            Minimum number of non-NaN observations required to produce a
            result.  Defaults to ``1`` so edge cycles are not dropped.

        Returns
        -------
        MSData
            New instance with smoothed data and the audit entry appended to
            ``ds.attrs["trace_corrections"]``.

        Raises
        ------
        ValueError
            If ``window`` is less than 2.
        KeyError
            If a specific ``block_id`` integer is not present in the dataset.
        """
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")

        if block_id is None or block_id == "all":
            target_blocks = self.block_ids
        else:
            if block_id not in self.block_ids:
                raise KeyError(f"Block {block_id} not found. Available: {self.block_ids}")
            target_blocks = [block_id]

        new_ds = self.ds.copy(deep=True)
        for bid in target_blocks:
            name = f"block_{bid}"
            smoothed = (
                new_ds[name]
                .rolling(cycle=window, center=center, min_periods=min_periods)
                .mean()
            )
            new_ds[name] = smoothed

        self._append_correction(
            new_ds,
            {
                "method": "smooth_trace_rolling",
                "window": window,
                "center": center,
                "min_periods": min_periods,
                "blocks": target_blocks,
            },
        )
        return MSData(ds=new_ds)

    def baseline_subtract(
        self,
        tos_start_seconds: float,
        tos_end_seconds: float,
        block_id: Union[int, Literal["all"], None] = "all",
    ) -> "MSData":
        """Subtract per-channel mean computed over a reference tos window.

        Parameters
        ----------
        tos_start_seconds : float
            Start of the baseline window in seconds (inclusive).
        tos_end_seconds : float
            End of the baseline window in seconds (inclusive).
        block_id : int, "all", or None, optional
            Which block(s) to correct.  ``"all"`` or ``None`` (default ``"all"``)
            processes every block.

        Returns
        -------
        MSData
            New instance with the per-channel baseline subtracted and the audit
            entry appended to ``ds.attrs["trace_corrections"]``.

        Raises
        ------
        ValueError
            If the ``tos`` coordinate is not set, or if the specified window
            contains no samples.
        KeyError
            If a specific ``block_id`` integer is not present in the dataset.
        """
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
        """Export a single block to a CSV file.

        Columns are labelled ``mz_<value>`` for block 0 or ``ch_<N>_<value>``
        for auxiliary blocks.  Optional ``timestamp`` and ``tos_s`` columns are
        prepended when available.

        Parameters
        ----------
        filepath : str or Path
            Destination file path.
        block_id : int, optional
            Block to export (default is the primary m/z block 0).

        Raises
        ------
        KeyError
            If ``block_id`` is not present in the dataset.
        """
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
        """Save the full dataset to a NetCDF4 file.

        The saved file can be reloaded with :meth:`from_netcdf`.

        Parameters
        ----------
        filepath : str or Path
            Destination file path.
        """
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
        """Construct an ``MSData`` directly from NumPy arrays.

        Parameters
        ----------
        cycle : numpy.ndarray
            1-D integer array of scan-cycle indices.
        channels : dict[int, numpy.ndarray]
            Mapping from block ID to 1-D channel coordinate array.  Block 0
            must be present and its values are interpreted as m/z in Da.
        values : dict[int, numpy.ndarray]
            Mapping from block ID to 2-D data array of shape
            ``(n_cycle, n_channels)``.  Must share the same keys as
            ``channels``.
        block_attrs : dict[int, dict[str, Any]] or None, optional
            Per-block attribute dicts (e.g. ``{"unit": "A", "type": "MID"}``).
        tos : numpy.ndarray or None, optional
            1-D float array of time-on-stream values in seconds, same length
            as ``cycle``.
        tos_start : str or pandas.Timestamp or None, optional
            Absolute start time corresponding to ``tos = 0``.
        ds_attrs : dict[str, Any] or None, optional
            Additional attributes to store on the dataset.

        Returns
        -------
        MSData
            New instance built from the supplied arrays.

        Raises
        ------
        ValueError
            If ``cycle`` is not 1-D, if ``channels`` and ``values`` have
            mismatched keys, if the primary block (id 0) is absent, if any
            value array has an incompatible shape, or if ``tos`` length
            differs from ``cycle``.
        """
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
        """Load an ``MSData`` from a Pfeiffer Quadstar ``.asc`` export.

        Accepts either a single file or a directory containing multiple
        ``.asc`` files from the same measurement run.  When a directory is
        given the files are concatenated in filesystem order and cycles are
        re-indexed as sequential integers.

        Parameters
        ----------
        filepath : str or Path
            Path to a ``.asc`` file or a directory of ``.asc`` files.
        tos_start : str or pandas.Timestamp or None, optional
            Absolute start time for the run.  When provided together with a
            ``Timestamp`` column in the data, time-on-stream is computed as
            elapsed seconds from this reference.  Accepts any string
            parseable by ``pandas.to_datetime``.
        drop_threshold_cols : bool, optional
            If ``True`` (default) threshold/qualifier columns are dropped from
            the parsed DataFrame before building the dataset.
        tz_str : str, optional
            Timezone name used to localise the parsed timestamps.  Default is
            ``"Europe/Amsterdam"``.

        Returns
        -------
        MSData
            New instance populated with the parsed data.

        Raises
        ------
        ValueError
            If no columns can be assigned to the primary m/z block.
        """

        filepath = Path(filepath)
        meta, df = quadstar.read_export(
            filepath, drop_threshold_cols=drop_threshold_cols, tz_str=tz_str
        )

        # For directory reads, column_map and datablocks come from the first
        # file's meta because all files in a run share the same channel layout.
        effective_meta = meta["file_metadata"][0] if "file_metadata" in meta else meta

        # ---- cycle axis ---------------------------------------------
        if not filepath.is_dir() and "Cycle" in df.columns:
            cycle = df["Cycle"].to_numpy(dtype=int)
        else:
            if filepath.is_dir():
                logger.info("Directory read — cycles re-indexed as sequential integers.")
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
        column_map = effective_meta.get("column_map", {})
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
        parser_blocks = dict(effective_meta.get("datablocks", {}))
        block_attrs: dict[int, dict[str, Any]] = {}
        for bid in chan_arrays:
            attrs = dict(parser_blocks.get(bid, {}))
            if bid != PRIMARY_BLOCK_ID and bid in label_dict:
                attrs["channel_labels"] = list(label_dict[bid])
            block_attrs[bid] = attrs

        # File-level metadata → ds.attrs (drop 'datablocks' and 'file_metadata').
        ds_attrs = {k: v for k, v in meta.items() if k not in ("datablocks", "file_metadata")}

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
        """Reload an ``MSData`` previously saved with :meth:`to_netcdf`.

        Parameters
        ----------
        filepath : str or Path
            Path to the NetCDF4 file.

        Returns
        -------
        MSData
            Instance reconstructed from the file.
        """
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
