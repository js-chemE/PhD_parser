import io
import re
import struct
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Union, Iterable, Dict, Any, Callable, Optional
import pandas as pd

import numpy as np
import requests

import logging

logger = logging.getLogger(__name__)

# =========================
# Basic utilities
# =========================

def _is_url(x: str) -> bool:
    return isinstance(x, str) and x.startswith(("http://", "https://"))


def _open_file(path: Union[str, Path]) -> io.BufferedReader:
    if _is_url(str(path)):
        r = requests.get(path, timeout=10)
        r.raise_for_status()
        return io.BytesIO(r.content)
    return open(Path(path), "rb")


def _read(fid: io.BufferedReader, dtype: str, count: int = 1) -> Union[int, float, np.ndarray]:
    fmt_map = {
        "uint8": "B",
        "uint32": "I",
        "int32": "i",
        "float32": "f",
    }
    fmt = "<" + fmt_map[dtype] * count
    size = struct.calcsize(fmt)

    data = fid.read(size)
    out = struct.unpack(fmt, data)

    return out[0] if count == 1 else np.array(out)


def _read_text(fid: io.BufferedReader, pos: int, size: int) -> str:
    fid.seek(pos)
    raw = fid.read(size)
    raw = re.sub(b"\x00+", b"\n", raw)
    return raw.decode("latin-1", errors="ignore").strip()

# =========================
# File Name Processing
# =========================

def extract_spectrum_id(path: Union[str, Path]) -> Optional[int]:
    """Extract a numeric spectrum identifier from a file path, trying all known strategies.

    Currently delegates to :func:`extract_spectrum_id_1`.  Returns ``None``
    and logs a warning if every strategy fails.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``.spa`` file whose stem is inspected.

    Returns
    -------
    int or None
        Extracted integer ID, or ``None`` if no strategy succeeds.
    """
    for extractor in [extract_spectrum_id_1]:
        try:
            return extractor(path)
        except Exception as e:
            logger.warning(f"Failed to extract spectra ID from {path} using {extractor.__name__}: {e}")
    return None

def extract_spectrum_id_1(path: Union[str, Path]) -> int:
    """Extract a spectrum index from a filename containing ``'Spectrum Index <N>'``.

    Returns ``0`` when the pattern is absent.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``.spa`` file whose stem is inspected.

    Returns
    -------
    int
        Integer parsed from ``'Spectrum Index <N>'``, or ``0`` if the pattern
        is not found.
    """
    name = Path(path).stem

    m = re.search(r"Spectrum Index (\d+)", name)
    if m:
        return int(m.group(1))

    return 0

def extract_spectrum_tos(path: Union[str, Path]) -> Optional[int]:
    """Extract elapsed time in seconds from a file path, trying all known strategies.

    Currently delegates to :func:`extract_spectrum_tos_1`.  Returns ``None``
    and logs a warning if every strategy fails.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``.spa`` file whose stem is inspected.

    Returns
    -------
    int or None
        Elapsed time in seconds, or ``None`` if no strategy succeeds.
    """
    for extractor in [extract_spectrum_tos_1]:
        try:
            return extractor(path)
        except Exception as e:
            logger.warning(f"Failed to extract spectra ID from {path} using {extractor.__name__}: {e}")
    return None

def extract_spectrum_tos_1(path: Union[str, Path]) -> int:
    """Extract elapsed time in seconds from a filename containing ``'at <H> Hours'``.

    Accepts both ``,`` and ``.`` as decimal separators.  Returns ``0`` when
    the pattern is absent.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``.spa`` file whose stem is inspected.

    Returns
    -------
    int
        Elapsed time in whole seconds, or ``0`` if the pattern is not found.
    """
    name = Path(path).stem

    # Match e.g. "2,46 Hours" or "2.46 Hours"
    m = re.search(r"at\s+([\d.,]+)\s*Hours", name)

    if m:
        hours_str = m.group(1).replace(",", ".")  # normalize decimal separator
        hours = float(hours_str)
        seconds = int(hours * 3600)
        return seconds

    return 0
# =========================
# Core SPA reader
# =========================

def _read_spa_single(path: Union[str, Path]) -> Dict[str, Any]:

    spectrum_id = extract_spectrum_id(path)
    spectrum_tos = extract_spectrum_tos(path)

    with _open_file(path) as fid:

        # ---- name ----
        name = _read_text(fid, 30, 256)

        # ---- timestamp ----
        fid.seek(296)
        t = _read(fid, "uint32")
        date = datetime(1899, 12, 31, tzinfo=timezone.utc) + timedelta(seconds=int(t))

        # ---- scan block ----
        pos = 304
        header: Dict[str, Any] = {}
        intensities: np.ndarray | None = None

        while True:
            fid.seek(pos)
            key = _read(fid, "uint8")

            if key == 2:
                fid.seek(pos + 2)
                hpos = _read(fid, "uint32")
                header = _read_header(fid, hpos)

            elif key == 3:
                intensities = _read_intensities(fid, pos)

            elif key in (0, 1):
                break

            pos += 16

        if intensities is None:
            logger.error(f"No intensity block found in {path}")
            raise ValueError(f"No spectral data found in {path}")

        # ---- x-axis ----
        x = np.linspace(header["firstx"], header["lastx"], int(header["nx"]))

        return {
            "name": name,
            "path": str(path),
            "datetime": date,
            "id": spectrum_id,
            "tos": spectrum_tos,
            "x": x,
            "xlabel": header["xtitle"],
            "xunit": header["xunit"],
            "v": intensities,
            "vunit": header["units"],
            "vlabel": header["title"],
        }


# =========================
# Header + intensities
# =========================

def _read_header(fid: io.BufferedReader, pos: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # ---- core values ----
    fid.seek(pos + 4)
    out["nx"] = _read(fid, "uint32")

    # ---- x units ----
    fid.seek(pos + 8)
    xkey = _read(fid, "uint8")
    xmap = {
        1: ("cm^-1", "wavenumber"),
        2: (None, "points"),
        3: ("nm", "wavelength"),
        4: ("um", "wavelength"),
        32: ("cm^-1", "raman shift"),
    }
    out["xunit"], out["xtitle"] = xmap.get(xkey, (None, "x"))

    # ---- y units ----
    fid.seek(pos + 12)
    ykey = _read(fid, "uint8")
    ymap = {
        11: ("%", "reflectance"),
        12: (None, "log(1/R)"),
        15: (None, "single beam"),
        16: ("%", "transmittance"),
        17: ("a.u.", "absorbance"),
        20: ("Kubelka_Munk", "Kubelka_Munk"),
        21: (None, "reflectance"),
        22: ("V", "detector signal"),
    }

    out_units, out_title = ymap.get(ykey)

    if out_title is None:
        logger.warning(f"Unknown y-axis key {ykey} in header at position {pos}. Defaulting to 'single beam' with no units.")
        out_units, out_title = (None, "single beam")

    out["units"], out["title"] = out_units, out_title

    # ---- x range ----
    fid.seek(pos + 16)
    out["firstx"] = _read(fid, "float32")

    fid.seek(pos + 20)
    out["lastx"] = _read(fid, "float32")

    return out


def _read_intensities(fid: io.BufferedReader, pos: int) -> np.ndarray:
    # ---- locate block ----
    fid.seek(pos + 2)
    ipos = _read(fid, "uint32")

    fid.seek(pos + 6)
    size = _read(fid, "uint32")

    n = size // 4

    # ---- read data ----
    fid.seek(ipos)
    return _read(fid, "float32", n)


# =========================
# Public API
# =========================

def read_spa(
    path: Union[str, Path, Iterable[Union[str, Path]]],
    sort_key: Optional[Callable[[Union[str, Path]], float]] = extract_spectrum_id,
    delta_time_seconds: Optional[float] = None,
    tos_start: Optional[pd.Timestamp | str] = None,
) -> Dict[str, Any]:
    """Read one or more Thermo OMNIC ``.spa`` files and return a standardised dict.

    Accepts a single file path, a directory (all ``.spa`` files are read), or
    an iterable of file paths.  Multiple spectra are stacked row-wise and their
    x-axes must be identical.

    The ``tos`` coordinate (elapsed seconds) is derived in order of priority:

    1. From ``tos_start`` — each spectrum's embedded timestamp minus
       ``tos_start``.
    2. From ``delta_time_seconds`` — uniform spacing ``0, dt, 2*dt, …``.
    3. Automatically from the first spectrum's timestamp (``tos[0] == 0``).

    Parameters
    ----------
    path : str, pathlib.Path, or iterable of str/Path
        Single file, directory, or sequence of files to read.
    sort_key : callable or None, optional
        Function applied to each file path to determine sort order when
        reading a directory or iterable.  Defaults to
        :func:`extract_spectrum_id`.  Pass ``None`` to skip sorting.
    delta_time_seconds : float or None, optional
        Fixed time step in seconds between consecutive spectra.  Mutually
        exclusive with ``tos_start`` (default is ``None``).
    tos_start : pandas.Timestamp or str or None, optional
        Absolute reference time used to convert embedded file timestamps to
        elapsed seconds.  Mutually exclusive with ``delta_time_seconds``
        (default is ``None``).

    Returns
    -------
    dict
        A dict with two keys:

        ``"data"``
            Dict containing ``"x"`` (1-D wavenumber array), ``"v"``
            (intensity array, shape ``(n_wavenumber,)`` for a single file or
            ``(n_spectra, n_wavenumber)`` for multiple files), and optionally
            ``"tos"`` (1-D elapsed-seconds array).

        ``"meta"``
            Dict of provenance metadata including ``"vlabel"``, ``"vunit"``,
            ``"xlabel"``, ``"xunit"``, ``"n_points"``, ``"min_x"``,
            ``"max_x"``, ``"name"``, ``"path"``, ``"datetime"``, and
            ``"tos_start"``.

    Raises
    ------
    ValueError
        If both ``delta_time_seconds`` and ``tos_start`` are supplied, if
        ``tos_start`` cannot be parsed, or if the x-axes of multiple spectra
        do not match.
    """
    if delta_time_seconds is not None and tos_start is not None:
        raise ValueError("Cannot specify both 'delta_time_seconds' and 'tos_start'. Please choose one method for calculating time of scan data.")

    if tos_start is not None:
        if isinstance(tos_start, str):
            try:
                tos_start = pd.to_datetime(tos_start)
            except Exception as e:
                logger.error(f"Failed to parse 'tos_start' string '{tos_start}': {e}")
                raise ValueError(f"Invalid 'tos_start' value: {tos_start}. Must be a pandas Timestamp or a string parseable by pandas.")
        elif not isinstance(tos_start, pd.Timestamp):
            raise ValueError(f"'tos_start' must be a pandas Timestamp or a string, but got {type(tos_start)}")


    # ---- normalize ----
    if isinstance(path, (str, Path)):
        p = Path(path)

        if p.is_dir():
            files = list({f.resolve() for f in p.glob("*.spa")})
        else:
            r = _read_spa_single(p)

            meta = {
                "vunit": r["vunit"],
                "vlabel": r["vlabel"],
                "xlabel": r["xlabel"],
                "xunit": r["xunit"],
                "n_points": len(r["x"]),
                "min_x": float(r["x"][0]),
                "max_x": float(r["x"][-1]),
                "name": [r["name"]],
                "path": [r["path"]],
                "datetime": [r["datetime"]],
            }

            return {
                "data": {
                    "x": r["x"],
                    "v": r["v"], #.reshape(1, -1),
                },
                "meta": meta,
            }

    else:
        files = [Path(p).resolve() for p in path]

    # ---- optional sorting ----
    if sort_key is not None:
        files = sorted(files, key=sort_key)

    # ---- read ----
    results = [_read_spa_single(f) for f in files]
    r0 = results[0]

    # ---- x axis ----
    x = results[0]["x"]

    # ---- consistency ----
    for r in results[1:]:
        if not np.allclose(r["x"], x):
            logger.error(f"X axes do not match between {results[0]['path']} and {r['path']}")
            raise ValueError("X axes do not match between spectra")

    # ---- stack ----
    v = np.vstack([r["v"] for r in results])

    logger.debug("Started reading TOS data for spectra.")
    if tos_start is not None:
        try:
            tos = np.asarray([(r["datetime"] - tos_start).total_seconds() for r in results])
            logger.debug(f"Calculated time offsets (tos) from 'tos_start'. tos_start={tos_start}, tos[0]={tos[0]}, tos[-1]={tos[-1]}")
        except Exception as e:
            logger.error(f"Error calculating time offsets from 'tos_start': {e}")
            tos = None

    elif delta_time_seconds is not None:
        tos = np.arange(v.shape[0]) * delta_time_seconds
        logger.debug(f"Calculated time offsets (tos) from 'delta_time_seconds'. delta_time_seconds={delta_time_seconds}, tos[0]={tos[0]}, tos[-1]={tos[-1]}")

    else:
        try:
            tos_start = r0["datetime"]
            tos = np.asarray([
                pd.to_timedelta(r["datetime"]- tos_start).total_seconds() for r in results
            ])
            logger.debug(f"Calculated time offsets (tos) from datetimes and first timestamp. tos_start={tos_start}, tos[0]={tos[0]}, tos[-1]={tos[-1]}")

        except Exception as e:
            logger.error(f"Error calculating time offsets: {e}")
            tos = None


    # ---- meta ----
    meta = {
        "vunit": r0["vunit"],
        "vlabel": r0["vlabel"],
        "xlabel": r0["xlabel"],
        "xunit": r0["xunit"],
        "n_points": len(x),
        "min_x": float(x[0]),
        "max_x": float(x[-1]),
        "name": [r["name"] for r in results],
        "path": [r["path"] for r in results],
        "datetime": [r["datetime"] for r in results],
        "tos_start": tos_start,
    }

    logger.info(f"Successfully read {len(results)} spectra from {len(files)} files. x points={len(x)}, v shape={v.shape}, tos shape={tos.shape if tos is not None else None}")
    return {
        "data": {
            "x": x,
            "tos": tos if tos is not None else None,
            "v": v,
        },
        "meta": meta,
    }
