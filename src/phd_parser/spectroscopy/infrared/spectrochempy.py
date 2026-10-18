"""Read Thermo OMNIC .spa files via the spectrochempy library.

Returns the same dict structure as :func:`phd_parser.spectroscopy.infrared.omnic.read_spa`
so :meth:`~phd_parser.spectroscopy.SpectroscopyData.from_omnic_spa` can use either backend
transparently.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd

from phd_parser.spectroscopy.infrared import omnic

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

    Drop-in replacement for :func:`phd_parser.spectroscopy.infrared.omnic.read_spa`:
    accepts the same arguments and returns the same dict structure
    (``"data"`` / ``"meta"``), so the caller does not need to know which
    backend was used.

    Elapsed-time (``tos``) values come from the acquisition timestamps
    SpectroChemPy reads out of each file, or from ``delta_time_seconds``;
    the hours encoded in OMNIC's file names are a last resort, with a
    warning, when no timestamp can be read at all.

    Parameters
    ----------
    path : str, pathlib.Path, or iterable of str/Path
        Single ``.spa`` file, directory of ``.spa`` files, or explicit list
        of file paths.
    sort_key : callable or None, optional
        Key function applied to each resolved ``Path`` to determine sort
        order (default: :func:`~phd_parser.spectroscopy.infrared.omnic.extract_spectrum_id`).
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
        Same structure as :func:`phd_parser.spectroscopy.infrared.omnic.read_spa`:

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

    # A single file is just a one-element series: same code path, same returned
    # structure, so tos/tos_start are applied either way.
    files = omnic.resolve_spa_files(path, sort_key)

    # ---- read via spectrochempy ----
    nd = scp.read_omnic(*files)

    # wavenumber: always cm⁻¹, ensure ascending
    x_cm = np.asarray(nd.x.data).ravel()
    values = np.asarray(nd.data)  # shape (n_files, n_wavenumber), always 2-D
    # SpectroChemPy returns the spectra in its own order (sorted by file name as text, so
    # "Index 10" comes before "Index 2"), not in the order the files were given. Put the
    # rows back into our order, which is the order of the timestamps below.
    row_order = _scp_row_order(nd, files)
    if row_order is not None:
        values = values[row_order]
    _verify_rows_against_files(files, x_cm, values)

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
        acquisition_times = _acquisition_times(nd, files, row_order)
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

    Accepts any ``NDDataset`` regardless of how it was created, from a file,
    from spectrochempy processing, or built programmatically, and returns the
    same ``{"data": ..., "meta": ...}`` structure as :func:`read_spa`, so the
    result feeds directly into :meth:`~phd_parser.spectroscopy.SpectroscopyData.from_spectrochempy`.

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


def _verify_rows_against_files(files: list[Path], x_cm: np.ndarray, values: np.ndarray, n_checks: int = 3) -> None:
    """Re-read a few files with the built-in parser and check that each row holds its own file's spectrum.

    The built-in reader takes data and metadata from one file handle, so its
    pairing cannot be wrong; comparing the first, middle and last row against
    it catches any reordering a library did on the way.

    Raises
    ------
    RuntimeError
        If a row does not match the spectrum stored in its file.
    """
    if len(files) < 2:
        return
    positions = sorted({0, len(files) // 2, len(files) - 1})[:n_checks]
    for position in positions:
        single = omnic._read_spa_single(files[position])
        x_single = np.asarray(single["x"], dtype=float).ravel()
        v_single = np.asarray(single["v"], dtype=float).ravel()
        # orient the file's axis like the library's (the caller may not have flipped yet)
        if x_single.size > 1 and x_cm.size > 1 and (x_single[0] > x_single[-1]) != (x_cm[0] > x_cm[-1]):
            x_single, v_single = x_single[::-1], v_single[::-1]
        # both axes come from the same float32 header values; SpectroChemPy rebuilds its own grid from them
        if x_single.size != x_cm.size or not np.allclose(x_single, x_cm, rtol=1e-5, atol=1e-2):
            raise RuntimeError(
                f"SpectroChemPy read {files[position].name} on a different wavenumber axis than the file holds."
            )
        # both readers decode the same float32 values, so the match is exact up to float64 rounding;
        # consecutive scans differ by far more than this
        if not np.allclose(values[position], v_single, rtol=1e-6, atol=1e-7, equal_nan=True):
            raise RuntimeError(
                f"The spectrum SpectroChemPy returned for {files[position].name} is not the one stored in that "
                "file: the library changed the order of the spectra and it could not be recovered. "
                "Read with backend='omnic' instead."
            )


def _scp_row_order(nd, files: list[Path]) -> Optional[list[int]]:
    """The row of ``nd.data`` that holds each of ``files``, read from SpectroChemPy's file-name labels.

    ``read_omnic`` labels its y axis with the source path of every spectrum
    (second label column); that is the only record of the order it used.
    Returns ``None`` when the labels carry no file names, in which case the
    rows are assumed to be in the given order.
    """
    labels = getattr(getattr(nd, "y", None), "labels", None)
    if labels is None:
        return None
    columns = np.atleast_2d(np.asarray(labels, dtype=object))
    if columns.shape[0] != len(files) and columns.shape[-1] == len(files):
        columns = columns.T
    for column in range(columns.shape[1]):
        names = [Path(str(item)).name for item in columns[:, column]]
        if all(name.lower().endswith(".spa") for name in names):
            row_of = {name: row for row, name in enumerate(names)}
            if len(row_of) == len(files) and all(f.name in row_of for f in files):
                return [row_of[f.name] for f in files]
    return None


def _acquisition_times(nd, files: list[Path], row_order: Optional[list[int]] = None) -> Optional[list[pd.Timestamp]]:
    """When each scan was actually recorded, in scan order.

    Prefers spectrochempy's own acquisition labels and falls back to the
    timestamp stored inside each ``.spa`` file, never to the filename, whose
    encoded hours are rounded to 36 s and restart at zero for every new series.
    """
    scp_times = _extract_scp_timestamps(nd)
    if scp_times is not None and len(scp_times) == len(files):
        return [scp_times[row] for row in row_order] if row_order is not None else scp_times

    try:
        return [pd.Timestamp(omnic.read_spa_datetime(f)) for f in files]
    except Exception as exc:  # noqa: BLE001, any unreadable file falls through
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
