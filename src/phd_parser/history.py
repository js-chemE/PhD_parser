"""Processing history: every step applied to a data object is recorded on the object itself.

A core class decorates its public, immutable methods with :func:`records_step`.
Each call appends one entry to ``ds.attrs["history"]`` of the *returned*
object (the input is never touched)::

    {"step": "smooth_savgol",
     "parameters": {"window_length": 21, "polyorder": 3},
     "time": "2026-10-15T09:12:03+00:00",
     "version": "0.1.0"}

Constructors (``from_omnic_spa``, ``from_ezchrom_rslt``, ...) are recorded the
same way, so the first entry says which files the data came from. Only the
outermost call is recorded: ``with_background_scan`` calls ``select_by_idx``
internally, and only ``with_background_scan`` shows up.

The history is plain JSON-compatible data. :func:`encode_attrs` and
:func:`decode_attrs` turn it (and any other nested attribute) into a JSON
string and back, so it survives a NetCDF round trip even with the NetCDF3
writer of scipy.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HISTORY_KEY = "history"
JSON_PREFIX = "json:"
_MAX_LISTED_VALUES = 20

_depth: contextvars.ContextVar[int] = contextvars.ContextVar("phd_parser_history_depth", default=0)

F = TypeVar("F", bound=Callable[..., Any])


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("phd_parser")
    except Exception:  # not installed as a distribution
        return "unknown"


_VERSION = _package_version()


def to_json_safe(value: Any) -> Any:
    """Turn a method argument into something JSON can store, keeping it readable.

    Numbers, strings, booleans and None stay; paths become strings, timestamps
    ISO strings; short sequences and arrays become lists, long ones a summary
    (shape, first and last value); data objects become their type, shape and
    own history; anything else its ``repr``, truncated.

    Parameters
    ----------
    value : Any
        The argument.

    Returns
    -------
    Any
        A JSON-compatible representation.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (np.ndarray, list, tuple)):
        array = np.asarray(value, dtype=object) if not isinstance(value, np.ndarray) else value
        if array.size <= _MAX_LISTED_VALUES:
            # small, also when nested ([[2220, 2420]]): kept as (nested) lists, so the step stays replayable
            return [to_json_safe(item) for item in (value.tolist() if isinstance(value, np.ndarray) else value)]
        flat = array.ravel()
        return {"array_shape": list(array.shape), "first": to_json_safe(flat[0]), "last": to_json_safe(flat[-1])}
    if hasattr(value, "ds") and hasattr(value.ds, "attrs"):
        return {
            "type": type(value).__name__,
            "sizes": {str(dim): int(size) for dim, size in value.ds.sizes.items()},
            HISTORY_KEY: list(getattr(value, "history", None) or get_history(value.ds.attrs)),
        }
    text = repr(value)
    return text if len(text) <= 200 else text[:197] + "..."


def make_entry(step: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Build one history entry.

    Parameters
    ----------
    step : str
        Name of the method.
    parameters : dict
        Its arguments, already JSON-safe.

    Returns
    -------
    dict
        ``{"step", "parameters", "time", "version"}``.
    """
    return {
        "step": step,
        "parameters": parameters,
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": _VERSION,
    }


def records_step(func: F) -> F:
    """Decorate a method so the object it returns carries one more history entry.

    Works on instance methods and, placed under ``@classmethod``, on
    constructors. Nested decorated calls are not recorded, only the outermost.
    A method that returns ``self`` unchanged records nothing.

    Parameters
    ----------
    func : callable
        Method returning a data object with a ``ds`` (``xr.Dataset``) field.

    Returns
    -------
    callable
        The wrapped method.
    """
    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        token = _depth.set(_depth.get() + 1)
        try:
            result = func(*args, **kwargs)
        finally:
            _depth.reset(token)
        if _depth.get() > 0:
            return result
        owner = args[0] if args else None
        if not (hasattr(result, "ds") and hasattr(result.ds, "attrs")) or result is owner:
            return result
        try:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            parameters = {
                name: to_json_safe(value)
                for name, value in list(bound.arguments.items())[1:]  # drop self / cls
            }
        except TypeError:
            parameters = {}
        return append_history(result, func.__name__, parameters, owner=owner)

    return wrapper  # type: ignore[return-value]


@contextlib.contextmanager
def suppress_recording():
    """Within this block decorated steps record nothing (for a caller that records the whole thing itself)."""
    token = _depth.set(_depth.get() + 1)
    try:
        yield
    finally:
        _depth.reset(token)


def append_history(obj: Any, step: str, parameters: dict[str, Any], owner: Any = None) -> Any:
    """Return ``obj`` with one more history entry (for steps a decorator cannot wrap).

    A class that keeps its attributes outside ``ds`` too (``LVData`` and its
    ``metadata``) defines ``_with_history(history)``; for every other class
    the history goes into ``ds.attrs``.

    Parameters
    ----------
    obj : data object
        Has a ``ds`` (``xr.Dataset``) field.
    step : str
        Name of the step.
    parameters : dict
        Its arguments, JSON-safe (see :func:`to_json_safe`).
    owner : data object, optional
        The input of the step. If ``obj`` shares its Dataset, the Dataset is
        copied first, so recording never touches the input.

    Returns
    -------
    data object
        ``obj`` or a copy of it, carrying the new entry.
    """
    previous = obj.history if isinstance(getattr(type(obj), "history", None), property) else get_history(obj.ds.attrs)
    history = [*previous, make_entry(step, parameters)]
    with_history = getattr(obj, "_with_history", None)
    if with_history is not None:
        return with_history(history)
    if owner is not None and getattr(owner, "ds", None) is obj.ds:
        obj = obj.model_copy(update={"ds": obj.ds.copy()})
    obj.ds.attrs = {**obj.ds.attrs, HISTORY_KEY: history}
    return obj


def get_history(ds_attrs: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the history stored in a dataset's attributes (decoding it if needed).

    Parameters
    ----------
    ds_attrs : dict
        ``ds.attrs`` of a data object.

    Returns
    -------
    list of dict
        The entries, oldest first; empty if none.
    """
    return list(decode_attr(ds_attrs.get(HISTORY_KEY, [])))


def encode_attr(value: Any) -> Any:
    """Make one attribute NetCDF-storable.

    Dicts, lists that are not purely numeric, booleans, ``None`` and
    non-ASCII text become a JSON string with the prefix ``"json:"``; numbers,
    ASCII strings and numeric lists stay as they are.
    """
    numeric_list = isinstance(value, (list, tuple)) and len(value) > 0 and all(
        isinstance(item, (int, float, np.integer, np.floating)) and not isinstance(item, bool) for item in value
    )
    if value is None or isinstance(value, (dict, bool, np.bool_)) \
            or (isinstance(value, (list, tuple)) and not numeric_list) \
            or (isinstance(value, str) and not value.isascii()):
        return JSON_PREFIX + json.dumps(to_json_safe(value))
    return value


def decode_attr(value: Any) -> Any:
    """Inverse of :func:`encode_attr`."""
    if isinstance(value, str) and value.startswith(JSON_PREFIX):
        return json.loads(value[len(JSON_PREFIX):])
    return value


def encode_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Apply :func:`encode_attr` to every attribute."""
    return {key: encode_attr(value) for key, value in attrs.items()}


def decode_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Apply :func:`decode_attr` to every attribute."""
    return {key: decode_attr(value) for key, value in attrs.items()}


def format_history(history: list[dict[str, Any]]) -> str:
    """Render a history as readable lines, one step per line.

    Parameters
    ----------
    history : list of dict
        As returned by :func:`get_history`.

    Returns
    -------
    str
        ``"1. 2026-10-15T09:12:03+00:00  smooth_savgol(window_length=21, polyorder=3)"`` ...
    """
    lines = []
    for number, entry in enumerate(history, start=1):
        parameters = ", ".join(
            f"{name}={json.dumps(value) if not isinstance(value, str) else repr(value)}"
            for name, value in entry.get("parameters", {}).items()
        )
        if len(parameters) > 160:
            parameters = parameters[:157] + "..."
        lines.append(f"{number}. {entry.get('time', '')}  {entry.get('step', '?')}({parameters})")
    return "\n".join(lines)


class HistoryMixin:
    """``history`` and ``format_history`` for a data class that keeps the history in ``ds.attrs``."""

    @property
    def history(self) -> list[dict[str, Any]]:
        """Every processing step that produced this object, oldest first.

        Each entry is ``{"step", "parameters", "time", "version"}``; the first
        one names the constructor and the file(s) read. Stored in
        ``ds.attrs["history"]`` and saved by ``to_netcdf``.

        Returns
        -------
        list of dict
            The entries; empty for an object built directly from a Dataset.
        """
        return get_history(self.ds.attrs)

    def format_history(self) -> str:
        """Return :attr:`history` as readable text, one numbered step per line."""
        return format_history(self.history)
