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
            ``"x"`` (cm⁻¹, ascending), ``"v"`` (1-D for single file or
            2-D ``(n_spectra, n_wavenumber)`` for multiple), optionally
            ``"tos"`` (elapsed seconds, 1-D).

        ``"meta"``
            ``"vlabel"``, ``"vunit"``, ``"xlabel"``, ``"xunit"``,
            ``"n_points"``, ``"min_x"``, ``"max_x"``, ``"name"``,
            ``"path"``, ``"tos_start"``.

    Raises
    ------
    ImportError
        If spectrochempy is not installed.
    ValueError
        If both ``delta_time_seconds`` and ``tos_start`` are supplied.
    """
    import spectrochempy as scp  # deferred so ImportError is explicit

    if delta_time_seconds is not None and tos_start is not None:
        raise ValueError("Specify either 'delta_time_seconds' or 'tos_start', not both.")

    # ---- resolve files ----
    if isinstance(path, (str, Path)):
        p = Path(path)
        if p.is_dir():
            files = list({f.resolve() for f in p.glob("*.spa")})
            single = False
        else:
            files = [p.resolve()]
            single = True
    else:
        files = [Path(f).resolve() for f in path]
        single = len(files) == 1

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

    if not single:
        if delta_time_seconds is not None:
            tos = np.arange(len(files), dtype=float) * delta_time_seconds
        else:
            # Try acquisition timestamps from spectrochempy first
            scp_times = _extract_scp_timestamps(nd)
            if scp_times is not None:
                if tos_start is not None:
                    resolved_tos_start = pd.Timestamp(tos_start)
                    tos = np.array(
                        [(t - resolved_tos_start).total_seconds() for t in scp_times],
                        dtype=float,
                    )
                else:
                    resolved_tos_start = scp_times[0]
                    tos = np.array(
                        [(t - resolved_tos_start).total_seconds() for t in scp_times],
                        dtype=float,
                    )
            else:
                # Fallback: derive tos from filename-encoded hours
                raw_tos = np.array(
                    [omnic.extract_spectrum_tos(f) or 0 for f in files], dtype=float
                )
                if tos_start is not None:
                    resolved_tos_start = pd.Timestamp(tos_start)
                    tos = raw_tos
                else:
                    tos = raw_tos - raw_tos[0]

    if single:
        values = values.squeeze()

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
