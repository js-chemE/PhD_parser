"""The time origin shared by every time-resolved class: ``tos`` in seconds plus one ``tos_start``.

``LVData``, ``SpectroscopyData``, ``MSData`` and ``GCData`` all store elapsed
seconds (``tos``) along their sample axis and one absolute origin,
``tos_start``, as an ISO string in ``ds.attrs``. :class:`TimeOriginMixin` is
the one implementation of everything that follows from that: the
``tos_start`` and ``timestamps`` properties, the four origin operations
``with_tos_start`` / ``set_tos_start`` / ``del_tos_start`` /
``move_tos_start_by`` and the wall-clock window ``select_timestamp_range``.

A host class provides ``ds`` (an ``xarray.Dataset`` with a ``tos``
coordinate), ``tos`` (the coordinate as an array or ``None``), ``_new(ds)``
(a copy around another Dataset) and ``select_tos_range(min_s, max_s,
empty=...)``. A host that keeps the origin somewhere else as well
(``LVData`` mirrors it into ``metadata``) overrides the two hooks
``_tos_start_raw`` and ``_with_tos_start``.
"""

from datetime import datetime
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from phd_parser._selection import EmptyWindow
from phd_parser.history import records_step

# Anything pandas.Timestamp can parse into an absolute point in time.
TimestampLike = Union[pd.Timestamp, datetime, np.datetime64, str]


def _to_timedelta(delta: Union[float, pd.Timedelta, str]) -> pd.Timedelta:
    """Read a plain number as seconds; hand anything else to pandas.Timedelta."""
    if isinstance(delta, (int, float, np.integer, np.floating)) and not isinstance(delta, bool):
        return pd.Timedelta(seconds=float(delta))
    return pd.Timedelta(delta)


def _to_timestamp(value: TimestampLike, reference: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """Parse a timestamp-like value and align its timezone with ``reference``.

    A naive value is read as wall-clock time in the reference's timezone; an
    aware value handed to a naive reference keeps its wall-clock reading and
    drops the offset. Both sides aware (or both naive) is left alone.
    """
    import logging

    timestamp = pd.Timestamp(value)
    if reference is None:
        return timestamp
    if timestamp.tzinfo is None and reference.tzinfo is not None:
        return timestamp.tz_localize(reference.tzinfo)
    if timestamp.tzinfo is not None and reference.tzinfo is None:
        logging.getLogger(__name__).warning(
            f"tos_start {reference} is timezone-naive but {timestamp} is not; "
            "comparing wall-clock times and ignoring the offset."
        )
        return timestamp.tz_localize(None)
    return timestamp


class TimeOriginMixin:
    """The ``tos_start`` family and the wall-clock window, for any class with ``tos`` and ``ds``.

    See the module docstring for what the host class provides.
    """

    # ---------------------------------------------------------------- hooks

    def _tos_start_raw(self) -> Any:
        """Where the origin is stored: ``ds.attrs["tos_start"]`` unless a host keeps it elsewhere."""
        return self.ds.attrs.get("tos_start")

    def _with_tos_start(self, ds: xr.Dataset, tos_start: Optional[pd.Timestamp]) -> Any:
        """A copy around ``ds`` with the origin set to ``tos_start`` (or removed for ``None``)."""
        attrs = dict(ds.attrs)
        if tos_start is None:
            attrs.pop("tos_start", None)
        else:
            attrs["tos_start"] = tos_start.isoformat()
        ds.attrs = attrs
        return self._new(ds)

    # ---------------------------------------------------------------- accessors

    @property
    def tos_start(self) -> Optional[pd.Timestamp]:
        """Absolute time of ``tos = 0``, or ``None`` if the data carry no origin."""
        raw = self._tos_start_raw()
        return None if raw is None else pd.Timestamp(raw)

    @property
    def timestamps(self) -> Optional[pd.DatetimeIndex]:
        """Absolute time of every sample, ``tos_start + tos``; ``None`` without ``tos`` or an origin."""
        tos = self.tos
        tos_start = self.tos_start
        if tos is None or tos_start is None:
            return None
        return pd.DatetimeIndex(tos_start + pd.to_timedelta(np.asarray(tos, dtype=float), unit="s"))

    # ---------------------------------------------------------------- the four operations

    @records_step
    def with_tos_start(self, tos_start: TimestampLike) -> Any:
        """Re-anchor ``tos`` to a new origin, leaving every absolute timestamp unchanged.

        Moves the zero of the ``tos`` axis: each value is shifted by minus the
        distance the origin moved, so ``tos_start + tos`` still resolves to the
        moment each sample was recorded. This is what you want when the
        experiment's reference point changes (e.g. aligning to when gas flow
        started rather than when the instrument did).

        Parameters
        ----------
        tos_start : pandas.Timestamp, datetime, numpy.datetime64 or str
            New absolute origin. A timezone-naive value is read as wall-clock
            time in the timezone of the current origin.

        Returns
        -------
        Same class as ``self``
            New instance with the new origin and rebased ``tos`` values. If no
            origin was set before, ``tos`` is left untouched and anchored to
            the new one.

        See Also
        --------
        set_tos_start : Replace the origin *without* touching ``tos``.
        move_tos_start_by : Move the origin by a relative amount.
        """
        old_tos_start = self.tos_start
        new_tos_start = _to_timestamp(tos_start, old_tos_start)
        ds = self.ds.copy()
        if old_tos_start is None:
            import logging

            logging.getLogger(type(self).__module__).info(
                f"No previous tos_start: keeping tos as-is and anchoring it to {new_tos_start}."
            )
        elif "tos" in ds.coords:
            shift_seconds = (new_tos_start - old_tos_start).total_seconds()
            ds = ds.assign_coords(tos=ds.coords["tos"] - shift_seconds)
        return self._with_tos_start(ds, new_tos_start)

    @records_step
    def set_tos_start(self, tos_start: TimestampLike) -> Any:
        """Replace the origin without touching ``tos``: every absolute timestamp moves.

        Use this to correct a wrong origin: the elapsed times are right, the
        wall clock they were anchored to was not.

        Parameters
        ----------
        tos_start : pandas.Timestamp, datetime, numpy.datetime64 or str
            New absolute origin. A timezone-naive value is read as wall-clock
            time in the timezone of the current origin.

        Returns
        -------
        Same class as ``self``
            New instance with the new origin and unchanged ``tos`` values.

        See Also
        --------
        with_tos_start : Re-anchor ``tos`` so the absolute timestamps survive.
        """
        return self._with_tos_start(self.ds.copy(), _to_timestamp(tos_start, self.tos_start))

    @records_step
    def del_tos_start(self) -> Any:
        """Drop the origin, keeping ``tos`` as a purely relative axis.

        Returns
        -------
        Same class as ``self``
            New instance without a ``tos_start``; ``timestamps`` becomes
            ``None``. An equivalent instance if none was set.
        """
        return self._with_tos_start(self.ds.copy(), None)

    @records_step
    def move_tos_start_by(self, delta: Union[float, pd.Timedelta, str]) -> Any:
        """Move the origin by a relative amount and re-anchor ``tos`` to it.

        Equivalent to ``with_tos_start(tos_start + delta)``: absolute
        timestamps never change, so moving the origin *later* by ``delta``
        makes every ``tos`` value *smaller* by ``delta``. Pass a negative
        ``delta`` to move the origin earlier and grow the ``tos`` values.

        Parameters
        ----------
        delta : float or pandas.Timedelta or str
            How far to move the origin. A plain number is read as seconds;
            anything else is passed to ``pandas.Timedelta`` (``"90s"``,
            ``"1h30min"``).

        Returns
        -------
        Same class as ``self``
            New instance with the moved origin and rebased ``tos`` values.

        Raises
        ------
        ValueError
            If no ``tos_start`` is set, so there is nothing to move.
        """
        if self.tos_start is None:
            raise ValueError("No tos_start to move. Anchor the data first with set_tos_start(...).")
        return self.with_tos_start(self.tos_start + _to_timedelta(delta))

    # ---------------------------------------------------------------- the wall-clock window

    @records_step
    def select_timestamp_range(
        self,
        start: Optional[TimestampLike] = None,
        end: Optional[TimestampLike] = None,
        empty: EmptyWindow = "warn",
    ) -> Any:
        """Return a new instance restricted to an absolute time window.

        The wall-clock counterpart of ``select_tos_range``: the bounds are
        converted against ``tos_start`` and handed to it, so the two never
        drift apart. Naive bounds are read as wall-clock time in the timezone
        of ``tos_start``.

        Parameters
        ----------
        start, end : pandas.Timestamp, datetime, numpy.datetime64, str or None, optional
            Inclusive bounds; ``None`` leaves that side open.
        empty : {"warn", "raise"}, optional
            What to do when nothing falls inside the window: log it and
            return the empty selection (default), or raise ``ValueError``.

        Returns
        -------
        Same class as ``self``
            The samples inside the window.

        Raises
        ------
        ValueError
            If no ``tos_start`` is set, so absolute times cannot be resolved,
            or if the window is empty and ``empty="raise"``.
        """
        origin = self.tos_start
        if origin is None:
            raise ValueError(
                "select_timestamp_range requires a tos_start. Anchor the data first with set_tos_start(...)."
            )
        lower = None if start is None else (_to_timestamp(start, origin) - origin).total_seconds()
        upper = None if end is None else (_to_timestamp(end, origin) - origin).total_seconds()
        return self.select_tos_range(lower, upper, empty=empty)
