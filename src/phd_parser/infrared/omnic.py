import io
import re
import struct
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Union, Iterable, Tuple, Dict, Any, Callable, Optional
import pandas as pd

from venv import logger

import numpy as np
import requests


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
    for extractor in [extract_spectrum_id_1]:
        try:
            return extractor(path)
        except Exception as e:
            logger.warning(f"Failed to extract spectra ID from {path} using {extractor.__name__}: {e}")
    return None

def extract_spectrum_id_1(path: Union[str, Path]) -> int:
    name = Path(path).stem

    m = re.search(r"Spectrum Index (\d+)", name)
    if m:
        return int(m.group(1))

    return 0

def extract_spectrum_tos(path: Union[str, Path]) -> Optional[int]:
    for extractor in [extract_spectrum_tos_1]:
        try:
            return extractor(path)
        except Exception as e:
            logger.warning(f"Failed to extract spectra ID from {path} using {extractor.__name__}: {e}")
    return None

def extract_spectrum_tos_1(path: Union[str, Path]) -> int:
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
            "v": intensities,
            "units": header["units"],
            "xlabel": header["xtitle"],
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
    out["xunits"], out["xtitle"] = xmap.get(xkey, (None, "x"))

    # ---- y units ----
    fid.seek(pos + 12)
    ykey = _read(fid, "uint8")
    ymap = {
        17: ("absorbance", "absorbance"),
        16: ("%", "transmittance"),
        11: ("%", "reflectance"),
        22: ("V", "signal"),
    }
    out["units"], out["title"] = ymap.get(ykey, (None, "intensity"))

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
    paths: Union[str, Path, Iterable[Union[str, Path]]],
    sort_key: Optional[Callable[[Union[str, Path]], float]] = extract_spectrum_id,
    delta_time_seconds: Optional[float] = None
) -> Dict[str, Any]:

    # ---- normalize ----
    if isinstance(paths, (str, Path)):
        p = Path(paths)

        if p.is_dir():
            files = list({f.resolve() for f in p.glob("*.spa")})
        else:
            r = _read_spa_single(p)

            meta = {
                "units": r["units"],
                "xlabel": r["xlabel"],
                "vlabel": r["vlabel"],
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
        files = [Path(p).resolve() for p in paths]

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

    if delta_time_seconds is not None:
        tos = np.arange(v.shape[0]) * delta_time_seconds
    else:
        try:
            tos = np.asarray([
                pd.to_timedelta(r["datetime"]- r0["datetime"]).total_seconds() for r in results
            ])
        except Exception as e:
            logger.error(f"Error calculating time offsets: {e}")
            tos = None

    # ---- meta ----
    meta = {
        "units": r0["units"],
        "xlabel": r0["xlabel"],
        "vlabel": r0["vlabel"],
        "n_points": len(x),
        "min_x": float(x[0]),
        "max_x": float(x[-1]),
        "name": [r["name"] for r in results],
        "path": [r["path"] for r in results],
        "datetime": [r["datetime"] for r in results],
    }

    return {
        "data": {
            "x": x,
            "tos": tos if tos is not None else None,
            "v": v,
        },
        "meta": meta,
    }