"""Read Thermo OMNIC .spa files via the spectrochempy library.

Returns the same dict structure as :func:`phd_parser.infrared.omnic.read_spa`
so :meth:`~phd_parser.infrared.IRData.from_omnic_spa` can use either backend
transparently.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd

from phd_parser.infrared import omnic

logger = logging.getLogger(__name__)

# Normalise spectrochempy dataset titles to the vlabel strings expected by
# _OMNIC_VLABEL_TO_DATA_TYPE in core.py.  Lookup is case-insensitive.
_SCP_TITLE_TO_VLABEL: dict[str, str] = {
    "absorbance": "absorbance",
    "transmittance": "transmittance",
    "reflectance": "reflectance",
    "single beam": "single beam",
    "kubelka-munk": "Kubelka_Munk",
    "kubelka_munk": "Kubelka_Munk",
    "log(1/r)": "log(1/R)",
}


def read_spa(
    path: Union[str, Path, Iterable[Union[str, Path]]],
    sort_key: Optional[Callable[[Path], float]] = omnic.extract_spectrum_id,
    delta_time_seconds: Optional[float] = None,
    tos_start: Optional[Union[pd.Timestamp, str]] = None,
) -> dict:
    """Read one or more Thermo OMNIC ``.spa`` files via spectrochempy.

    Drop-in replacement for :func:`phd_parser.infrared.omnic.read_spa`:
    accepts the same arguments and returns the same dict structure
    (``"data"`` / ``"meta"``), so the caller does not need to know which
    backend was used.

    Elapsed-time (tos) values are derived from filenames via
    :func:`~phd_parser.infrared.omnic.extract_spectrum_tos` when no explicit
    timing argument is supplied.  Embedded file timestamps (used by the
    manual omnic parser) are not exposed by spectrochempy, so spectrochempy
    timestamps are attempted first and the filename fallback is used when they
    are unavailable.

    Parameters
    ----------
    path : str, pathlib.Path, or iterable of str/Path
        Single ``.spa`` file, directory of ``.spa`` files, or explicit list
        of file paths.
    sort_key : callable or None, optional
        Key function applied to each resolved ``Path`` to determine sort
        order (default: :func:`~phd_parser.infrared.omnic.extract_spectrum_id`).
        Pass ``None`` to skip sorting.
    delta_time_seconds : float or None, optional
        Fixed time step in seconds between consecutive spectra.  Mutually
        exclusive with ``tos_start``.
    tos_start : pandas.Timestamp or str or None, optional
        Absolute start time used to anchor tos values extracted from file
        metadata.  Mutually exclusive with ``delta_time_seconds``.

    Returns
    -------
    dict
        Same structure as :func:`phd_parser.infrared.omnic.read_spa`:

        ``"data"``
            ``"x"`` (cm⁻¹, ascending), ``"v"`` (2-D
            ``(n_spectra, n_wavenumber)``, one row per file), ``"tos"``
            (elapsed seconds, 1-D).

        ``"meta"``
            ``"vlabel"``, ``"vunit"``, ``"xlabel"``, ``"xunit"``,
            ``"n_points"``, ``"min_x"``, ``"max_x"``, ``"name"``,
            ``"path"``, ``"tos_start"``.

    Raises
    ------
    ImportError
        If spectrochempy is not installed.
    FileNotFoundError
        If *path* does not exist, or resolves to no ``.spa`` file at all.
    ValueError
        If both ``delta_time_seconds`` and ``tos_start`` are supplied.
    """
    import spectrochempy as scp  # deferred so ImportError is explicit

    if delta_time_seconds is not None and tos_start is not None:
        raise ValueError("Specify either 'delta_time_seconds' or 'tos_start', not both.")

    # ---- resolve files ----
    # A single file is just a one-element series: same code path, same returned
    # structure, so tos/tos_start are applied either way.
    if isinstance(path, (str, Path)):
        p = Path(path)
        if p.is_dir():
            files = list({f.resolve() for f in p.glob("*.spa")})
            if not files:
                raise FileNotFoundError(f"No .spa files found in directory {p}")
        else:
            if not p.exists():
                raise FileNotFoundError(p)
            files = [p.resolve()]
    else:
        files = [Path(f).resolve() for f in path]
        if not files:
            raise FileNotFoundError("No .spa files given: the supplied path list is empty")

    if sort_key is not None and len(files) > 1:
        files = sorted(files, key=sort_key)

    # ---- read via spectrochempy ----
    nd = scp.read_omnic(*files)

    # wavenumber: always cm⁻¹, ensure ascending
    x_cm = np.asarray(nd.x.data).ravel()
    values = np.asarray(nd.data)  # shape (n_files, n_wavenumber) — always 2-D

    if x_cm[0] > x_cm[-1]:
        x_cm = x_cm[::-1]
        values = values[:, ::-1]

    # ---- vlabel (for data_type lookup in core.py) ----
    title = (nd.title or "").strip()
    vlabel = _SCP_TITLE_TO_VLABEL.get(title.lower(), title or "absorbance")

    # ---- tos ----
    tos: Optional[np.ndarray] = None
    resolved_tos_start: Optional[pd.Timestamp] = None

    if delta_time_seconds is not None:
        tos = np.arange(len(files), dtype=float) * delta_time_seconds
        resolved_tos_start = pd.Timestamp(tos_start) if tos_start is not None else None
    else:
        acquisition_times = _acquisition_times(nd, files)
        if acquisition_times is not None:
            resolved_tos_start = (
                pd.Timestamp(tos_start) if tos_start is not None else acquisition_times[0]
            )
            tos = _elapsed_seconds(acquisition_times, resolved_tos_start)
        else:
            # Last resort: the hours encoded in the filenames. Rounded to two
            # decimals (0.01 h = 36 s) and restarted at zero on every OMNIC
            # restart, so this cannot place two series on a common axis.
            logger.warning(
                "No acquisition timestamps available for %d file(s); falling back to the hours "
                "encoded in the filenames, which are rounded to 36 s and restart at zero for "
                "every new series. Do not rely on this axis to merge separate measurements.",
                len(files),
            )
            raw_tos = np.array(
                [omnic.extract_spectrum_tos(f) or 0 for f in files], dtype=float
            )
            tos = raw_tos - raw_tos[0]
            if tos_start is not None:
                resolved_tos_start = pd.Timestamp(tos_start)

    meta: dict = {
        "vlabel": vlabel,
        "vunit": str(nd.units) if nd.units is not None else None,
        "xlabel": "wavenumber",
        "xunit": "cm^-1",
        "n_points": int(x_cm.size),
        "min_x": float(x_cm.min()),
        "max_x": float(x_cm.max()),
        "name": [f.name for f in files],
        "path": [str(f) for f in files],
        "tos_start": resolved_tos_start,
    }

    result: dict = {"data": {"x": x_cm, "v": values}, "meta": meta}
    if tos is not None:
        result["data"]["tos"] = tos
    return result


def read_nddataset(
    nd,
    delta_time_seconds: Optional[float] = None,
    tos_start: Optional[Union[pd.Timestamp, str]] = None,
) -> dict:
    """Convert a spectrochempy ``NDDataset`` to the standard parser dict.

    Accepts any ``NDDataset`` regardless of how it was created — from a file,
    from spectrochempy processing, or built programmatically — and returns the
    same ``{"data": ..., "meta": ...}`` structure as :func:`read_spa`, so the
    result feeds directly into :meth:`~phd_parser.infrared.IRData.from_scp`.

    Parameters
    ----------
    nd : spectrochempy.NDDataset
        Dataset to convert.  ``nd.x`` must be the wavenumber axis (cm⁻¹) and
        ``nd.data`` must be 2-D ``(n_scans, n_wavenumber)``.
    delta_time_seconds : float or None, optional
        Fixed time step in seconds between consecutive scans.  Mutually
        exclusive with ``tos_start``.
    tos_start : pandas.Timestamp or str or None, optional
        Absolute start time used to anchor elapsed-time values extracted from
        the dataset's y-axis timestamps.  Mutually exclusive with
        ``delta_time_seconds``.

    Returns
    -------
    dict
        ``"data"`` → ``"x"`` (cm⁻¹, ascending), ``"v"``
        (2-D ``(n_scans, n_wavenumber)``, also for a single scan), optionally
        ``"tos"`` (elapsed seconds).  ``"meta"`` → ``"vlabel"``, ``"vunit"``,
        ``"xlabel"``, ``"xunit"``, ``"n_points"``, ``"min_x"``, ``"max_x"``,
        ``"tos_start"``.

    Raises
    ------
    ValueError
        If both ``delta_time_seconds`` and ``tos_start`` are supplied.
    """
    if delta_time_seconds is not None and tos_start is not None:
        raise ValueError("Specify either 'delta_time_seconds' or 'tos_start', not both.")

    # wavenumber: cm⁻¹, ensure ascending
    x_cm = np.asarray(nd.x.data).ravel()
    values = np.asarray(nd.data)  # always 2-D (n_scans, n_wavenumber) from spectrochempy

    if x_cm[0] > x_cm[-1]:
        x_cm = x_cm[::-1]
        values = values[:, ::-1]

    n_scans = values.shape[0]

    # vlabel
    title = (nd.title or "").strip()
    vlabel = _SCP_TITLE_TO_VLABEL.get(title.lower(), title or "absorbance")

    # tos: spectrochempy timestamps only (no filename fallback here).
    # A one-scan dataset is treated like any other series, so it gets a tos too.
    tos: Optional[np.ndarray] = None
    resolved_tos_start: Optional[pd.Timestamp] = None

    if delta_time_seconds is not None:
        tos = np.arange(n_scans, dtype=float) * delta_time_seconds
    else:
        scp_times = _extract_scp_timestamps(nd)
        if scp_times is not None:
            t0 = pd.Timestamp(tos_start) if tos_start is not None else scp_times[0]
            resolved_tos_start = t0
            tos = np.array([(t - t0).total_seconds() for t in scp_times], dtype=float)
        elif tos_start is not None:
            resolved_tos_start = pd.Timestamp(tos_start)

    if tos is None and n_scans == 1:
        # A lone scan has no spacing to infer: it simply sits at the origin.
        tos = np.zeros(1, dtype=float)

    meta: dict = {
        "vlabel": vlabel,
        "vunit": str(nd.units) if nd.units is not None else None,
        "xlabel": "wavenumber",
        "xunit": "cm^-1",
        "n_points": int(x_cm.size),
        "min_x": float(x_cm.min()),
        "max_x": float(x_cm.max()),
        "tos_start": resolved_tos_start,
    }

    result: dict = {"data": {"x": x_cm, "v": values}, "meta": meta}
    if tos is not None:
        result["data"]["tos"] = tos
    return result


def _acquisition_times(nd, files: list[Path]) -> Optional[list[pd.Timestamp]]:
    """When each scan was actually recorded, in scan order.

    Prefers spectrochempy's own acquisition labels and falls back to the
    timestamp stored inside each ``.spa`` file — never to the filename, whose
    encoded hours are rounded to 36 s and restart at zero for every new series.
    """
    scp_times = _extract_scp_timestamps(nd)
    if scp_times is not None and len(scp_times) == len(files):
        return scp_times

    try:
        return [pd.Timestamp(omnic.read_spa_datetime(f)) for f in files]
    except Exception as exc:  # noqa: BLE001 — any unreadable file falls through
        logger.debug("Could not read acquisition timestamps from the .spa files: %s", exc)
        return None


def _elapsed_seconds(times: list[pd.Timestamp], origin: pd.Timestamp) -> np.ndarray:
    """Seconds from origin to each timestamp, with a clear error on tz mismatch."""
    if (origin.tzinfo is None) != (times[0].tzinfo is None):
        raise ValueError(
            f"Cannot anchor the acquisition times ({times[0]}) to tos_start ({origin}): one is "
            "timezone-aware and the other is not. Pass a timezone-aware tos_start, e.g. "
            'pd.Timestamp("2026-07-23 10:00", tz="Europe/Amsterdam").'
        )
    return np.array([(t - origin).total_seconds() for t in times], dtype=float)


def _extract_scp_timestamps(nd) -> Optional[list[pd.Timestamp]]:
    """Try to pull per-scan acquisition datetimes out of a spectrochempy NDDataset.

    Returns a list of ``pd.Timestamp`` in scan order, or ``None`` when no
    datetime information is accessible.
    """
    # spectrochempy stores acquisition times as labels on the y-axis coordinate
    try:
        y = nd.y
        if y is None:
            return None
        labels = getattr(y, "labels", None)
        if labels is None:
            return None
        flat = np.asarray(labels).ravel()
        if flat.size == 0:
            return None
        return [pd.Timestamp(lbl) for lbl in flat]
    except Exception as exc:
        logger.debug("Could not extract acquisition timestamps from spectrochempy NDDataset: %s", exc)
        return None
