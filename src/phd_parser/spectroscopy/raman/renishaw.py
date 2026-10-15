from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, Literal
import pandas as pd
import numpy as np

import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import IntEnum

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

__all__ = ["read_export_txt", "read_export_wdf", "WDFResult"]

# def _convert_value(value: str) -> Any:
#     try:
#         return float(value)
#     except ValueError:
#         return value

# ================================================
# wdf export format (WiRe)
# ================================================

# Enumerations  (from SpectroChemPy / py-wdf-reader)

class MeasurementType(IntEnum):
    Unspecified = 0
    Single      = 1
    Series      = 2
    Mapping     = 3

    def __str__(self):
        return self.name


class ScanType(IntEnum):
    Unspecified    = 0
    Static         = 1
    Continuous     = 2
    StepRepeat     = 3
    FilterScan     = 4
    FilterImage    = 5
    StreamLine     = 6
    StreamLineHR   = 7
    PointDetector  = 8

    def __str__(self):
        return self.name


class UnitType(IntEnum):
    """Physical unit codes used in Renishaw WDF files.

    Each member maps to a numeric identifier stored in the binary WDF header.
    Helper methods convert a member to a human-readable label, an SI
    conversion factor, or a physical dimension string.
    """

    Arbitrary        = 0
    RamanShift       = 1   # cm-1
    Wavelength       = 2   # nm
    Nanometre        = 3
    ElectronVolt     = 4
    Micron           = 5
    Counts           = 6
    Electrons        = 7
    Millimetres      = 8
    Metres           = 9
    Kelvin           = 10
    Pascal           = 11
    Seconds          = 12
    Milliseconds     = 13
    Hours            = 14
    Days             = 15
    Pixels           = 16
    Intensity        = 17
    RelativeIntensity= 18
    Degrees          = 19
    Radians          = 20
    Celsius          = 21
    Fahrenheit       = 22
    KelvinPerMinute  = 23
    AcquisitionTime  = 24
    Microseconds     = 25

    def label(self):
        """Return a human-readable unit label string.

        Returns
        -------
        str
            Unicode unit label (e.g. ``"cm⁻¹"``, ``"nm"``, ``"K"``).
            Falls back to the enum member name for unmapped values.
        """
        _map = {
            1: "cm⁻¹", 2: "nm", 3: "nm", 4: "eV", 5: "µm",
            8: "mm", 9: "m", 10: "K", 11: "Pa", 12: "s",
            13: "ms", 19: "°", 20: "rad", 21: "°C",
        }
        return _map.get(self.value, self.name)

    def si_factor(self):
        """Return the multiplicative factor that converts this unit to SI.

        Returns
        -------
        float
            Factor such that ``value_in_unit * si_factor()`` gives the value
            in the corresponding SI base unit.  Returns ``1`` for unmapped
            or dimensionless units.
        """
        _map = {
            1: 1e2, 2: 1e-9, 3: 1e-9, 4: 1.60218e-19, 5: 1e-6,
            8: 1e-3, 9: 1, 10: 1, 11: 1, 12: 1,
            13: 1e-3, 19: 1, 20: 1, 21: 1,
        }
        return _map.get(self.value, 1)

    def dimension(self):
        """Return the physical dimension string for this unit.

        Returns
        -------
        str
            Dimension name (e.g. ``"length⁻¹"``, ``"energy"``,
            ``"temperature"``).  Falls back to the enum member name for
            unmapped values.
        """
        _map = {
            1: "length⁻¹", 2: "length", 3: "length", 4: "energy", 5: "length",
            8: "length", 9: "length", 10: "temperature", 11: "pressure", 12: "time",
            13: "time", 19: "angle", 20: "angle", 21: "temperature",
        }
        return _map.get(self.value, self.name)


class DataType(IntEnum):
    Arbitrary   = 0
    RamanShift  = 1
    Intensity   = 2
    X           = 3
    Y           = 4
    Z           = 5
    Temperature = 9
    Pressure    = 10
    Time        = 11
    ElapsedTime = 18
    Checksum    = 16
    Flags       = 17

    def __str__(self): return self.name


class MapAreaType(IntEnum):
    Unspecified       = 0
    RandomPoints      = 1
    ColumnMajor       = 2
    Alternating       = 4
    LineFocusMapping  = 8
    SurfaceProfile    = 64
    XYLine            = 128



# Block-layout constants  (byte offsets within WDF1 header)

class _Off(IntEnum):
    # Generic block header
    block_name  = 0x00   # 4 bytes ascii
    block_uid   = 0x04   # int32
    block_size  = 0x08   # int64  (note: MATLAB used int32 — this is the spec)
    block_data  = 0x10   # first data byte after the 16-byte header

    # Within WDF1 block
    meas_info   = 0x3C
    spectral    = 0x98
    file_info   = 0xD0
    usr_name    = 0xF0
    wdf1_end    = 0x200  # WDF1 block is always 512 bytes

    # Within ORGN block
    origin_info = 0x14
    origin_incr = 0x18   # fixed header part before the data array

    # Within WMAP block
    wmap_origin = 0x10

    # Within WHTL block
    jpeg_header = 0x10



# Result dataclass

@dataclass
class WDFResult(BaseException):
    """Container for all data extracted from a Renishaw ``.wdf`` file.

    Populated by :func:`read_export_wdf`.  Fields cover core spectral data,
    measurement geometry, instrumental parameters, optional stage coordinates,
    and provenance metadata.
    """

    # Core spectral data
    wavenumber:       np.ndarray = field(default_factory=lambda: np.array([]))
    data:             np.ndarray = field(default_factory=lambda: np.array([]))

    # Measurement geometry
    measurement_type: MeasurementType = MeasurementType.Unspecified
    scan_type:        ScanType        = ScanType.Unspecified
    height:           int = 1
    width:            int = 1

    # Instrumental
    laser_cm1:   float = 0.0
    x_unit:      UnitType = UnitType.RamanShift
    data_unit:   UnitType = UnitType.Counts

    # Origin coordinates (from ORGN block, if present)
    x_pos:  Optional[np.ndarray] = None   # stage X positions
    y_pos:  Optional[np.ndarray] = None   # stage Y positions
    z_pos:  Optional[np.ndarray] = None   # stage Z positions
    times:  Optional[np.ndarray] = None   # timestamps (float seconds from start)

    # Metadata
    title:     str = ""
    username:  str = ""
    filename:  str = ""
    acq_time:  Optional[datetime] = None

    # Everything else parsed from the header
    meta: dict = field(default_factory=dict)


# =============================================================================
# Low-level helpers
# =============================================================================

def _u(fid, fmt: str, size: int):
    """Read and unpack a single value; `fmt` is a struct format character."""
    nbytes = size
    raw = fid.read(nbytes)
    if len(raw) < nbytes:
        raise EOFError("Unexpected end of file")
    return struct.unpack("<" + fmt, raw)[0]

def _read_int32(fid):  return _u(fid, "i", 4)
def _read_uint32(fid): return _u(fid, "I", 4)
def _read_int64(fid):  return _u(fid, "q", 8)
def _read_uint64(fid): return _u(fid, "Q", 8)
def _read_float(fid):  return _u(fid, "f", 4)
def _read_double(fid): return _u(fid, "d", 8)
def _read_utf8(fid, n): return fid.read(n).decode("utf-8", errors="replace").rstrip("\x00")

def _windows_time_to_datetime(win_time: int) -> datetime:
    """Convert Windows FILETIME (100-ns intervals since 1601-01-01) to datetime."""
    EPOCH_DIFF = 116_444_736_000_000_000  # 100-ns ticks between 1601 and 1970
    unix_us = (win_time - EPOCH_DIFF) / 10  # microseconds
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=unix_us)


# =============================================================================
# Block locator
# =============================================================================

def _locate_blocks(fid) -> dict[str, tuple[int, int, int]]:
    """
    Scan the file and build a map of  block_name -> (uid, start_pos, size).
    Block header layout: 4-char name | int32 uid | int64 size
    """
    blocks: dict[str, tuple[int, int, int]] = {}
    pos = 0
    while True:
        fid.seek(pos)
        raw = fid.read(16)
        if len(raw) < 16:
            break
        try:
            name = raw[:4].decode("ascii")
        except UnicodeDecodeError:
            break
        uid  = struct.unpack_from("<i", raw, 4)[0]
        size = struct.unpack_from("<q", raw, 8)[0]
        if size <= 0:
            break
        blocks[name] = (uid, pos, size)
        pos += size
    return blocks


# =============================================================================
# Block parsers
# =============================================================================

def _parse_wdf1(fid, blocks: dict, result: WDFResult):
    """Parse the WDF1 header block."""
    _, pos, _ = blocks["WDF1"]
    fid.seek(pos + _Off.meas_info)

    result.meta["point_per_spectrum"] = _read_int32(fid)
    result.meta["capacity"]           = _read_uint64(fid)
    result.meta["count"]              = _read_uint64(fid)
    result.meta["accumulation_count"] = _read_int32(fid)
    result.meta["y_size"]             = _read_int32(fid)  # XLST length
    result.meta["x_size"]             = _read_int32(fid)  # XLST length (alt)
    result.meta["other_data_count"]   = _read_int32(fid)

    app_name    = _read_utf8(fid, 24)
    app_version = [str(_u(fid, "H", 2)) for _ in range(4)]
    result.meta["application"] = f"{app_name} {'.'.join(app_version)}"

    result.scan_type        = ScanType(_read_int32(fid))
    result.measurement_type = MeasurementType(_read_int32(fid))

    fid.seek(pos + _Off.spectral)
    result.data_unit  = UnitType(_read_int32(fid))
    result.laser_cm1  = _read_float(fid)

    fid.seek(pos + _Off.file_info)
    result.username = _read_utf8(fid, _Off.usr_name - _Off.file_info)

    fid.seek(pos + _Off.usr_name)
    result.title = _read_utf8(fid, _Off.wdf1_end - _Off.usr_name)


def _parse_xlst(fid, blocks: dict, result: WDFResult):
    """Parse the XLST block (wavenumber / x-axis)."""
    _, pos, _ = blocks["XLST"]
    fid.seek(pos + _Off.block_data)
    datatype = DataType(_read_int32(fid))
    result.x_unit = UnitType(_read_int32(fid))
    n = result.meta.get("x_size") or result.meta.get("point_per_spectrum", 0)
    raw = fid.read(n * 4)
    result.wavenumber = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    result.datatype = datatype


def _parse_data(fid, blocks: dict, result: WDFResult):
    """Parse the DATA block."""
    _, pos, _ = blocks["DATA"]
    fid.seek(pos + _Off.block_data)
    count  = result.meta["count"]
    points = result.meta["point_per_spectrum"]
    raw    = fid.read(count * points * 4)
    result.data = np.frombuffer(raw, dtype="<f4").astype(np.float64)


def _parse_orgn(fid, blocks: dict, result: WDFResult):
    """Parse the ORGN block: stage coordinates and timestamps."""
    if "ORGN" not in blocks:
        return

    _, pos, _ = blocks["ORGN"]
    count    = result.meta["count"]
    n_other  = result.meta.get("other_data_count", 0)
    capacity = result.meta["capacity"]

    # Each row: int32 type_flag | int32 unit | 16-byte utf8 title | capacity × double
    row_size = _Off.origin_incr + 8 * capacity

    x_vals = y_vals = z_vals = times = None

    curpos = pos + _Off.origin_info
    for _ in range(n_other):
        fid.seek(curpos)
        p1       = _read_uint32(fid)
        datatype_int = p1 & ~(0b1 << 31)
        try:
            datatype = DataType(datatype_int)
        except ValueError:
            curpos += row_size
            continue

        # Skip checksums / flags
        if datatype in (DataType.Checksum, DataType.Flags):
            curpos += row_size
            continue

        unit_int = _read_uint32(fid)
        _read_utf8(fid, 0x10)  # annotation (16 bytes)

        if datatype == DataType.Time:
            vals = np.array([_read_uint64(fid) for _ in range(count)])
            try:
                dts = [_windows_time_to_datetime(v) for v in vals]
                result.acq_time = dts[0]
                t0 = dts[0]
                result.times = np.array([(d - t0).total_seconds() for d in dts])
            except Exception:
                pass
        else:
            vals = np.array([_read_double(fid) for _ in range(count)])
            if datatype == DataType.X:
                x_vals = vals
            elif datatype == DataType.Y:
                y_vals = vals
            elif datatype == DataType.Z:
                z_vals = vals

        curpos += row_size

    result.x_pos = x_vals
    result.y_pos = y_vals
    result.z_pos = z_vals


def _parse_wmap(fid, blocks: dict, result: WDFResult):
    """Parse the WMAP block to get map width/height and area type."""
    if "WMAP" not in blocks:
        return

    _, pos, _ = blocks["WMAP"]
    fid.seek(pos + _Off.wmap_origin)

    map_area_type = MapAreaType(_read_int32(fid))
    _read_int32(fid)          # unknown
    x_offset    = _read_float(fid)
    y_offset    = _read_float(fid)
    z_offset    = _read_float(fid)
    x_step      = _read_float(fid)
    y_step      = _read_float(fid)
    z_step      = _read_float(fid)
    x_size      = _read_int32(fid)
    y_size      = _read_int32(fid)

    result.width  = x_size
    result.height = y_size
    result.meta["map_area_type"] = map_area_type
    result.meta["map_x_offset"]  = x_offset
    result.meta["map_y_offset"]  = y_offset
    result.meta["map_x_step"]    = x_step
    result.meta["map_y_step"]    = y_step


def _reshape_data(result: WDFResult):
    """Reshape the flat data array into (height, width, n_wn) or (n_spectra, n_wn)."""
    count  = result.meta["count"]
    points = result.meta["point_per_spectrum"]
    h, w   = result.height, result.width

    if result.measurement_type == MeasurementType.Mapping and h > 1 and w > 1:
        if h * w == count:
            result.data = result.data.reshape(h, w, points)
        else:
            # Fallback: best-effort reshape as flat series
            result.data = result.data.reshape(count, points)
    elif count > 1:
        result.data = result.data.reshape(count, points)
    else:
        result.data = result.data.reshape(points)


# =============================================================================
# Main entry point
# =============================================================================

def read_export_wdf(filename: str | Path) -> WDFResult:
    """Parse a Renishaw WiRe binary WDF file and return all extracted data.

    Reads the WDF1 header, XLST (wavenumber axis), DATA, and optional ORGN
    (stage coordinates / timestamps) and WMAP (map geometry) blocks.  The
    flat data array is reshaped to ``(height, width, n_spectral)`` for maps
    or ``(n_spectra, n_spectral)`` for series before being returned.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the ``.wdf`` binary file.

    Returns
    -------
    WDFResult
        Dataclass populated with the spectral data, measurement geometry,
        instrumental parameters, optional stage coordinates, and metadata
        parsed from the file.

    Raises
    ------
    ValueError
        When one or more of the required blocks (``WDF1``, ``DATA``,
        ``XLST``) are absent from the file.
    """
    if not isinstance(filename, Path):
        filename = Path(filename)

    result = WDFResult(filename=str(filename))

    with open(filename, "rb") as fid:
        blocks = _locate_blocks(fid)

        required = {"WDF1", "DATA", "XLST"}
        missing  = required - blocks.keys()
        if missing:
            raise ValueError(f"Required blocks missing from file: {missing}")

        _parse_wdf1(fid, blocks, result)
        _parse_xlst(fid, blocks, result)
        _parse_data(fid, blocks, result)
        _parse_orgn(fid, blocks, result)   # optional, won't crash if absent
        _parse_wmap(fid, blocks, result)   # optional

    _reshape_data(result)

    return result

# ================================================
# txt export format
# ================================================

def read_export_txt(file_path: str | Path) -> Dict[str, Any]:
    """Parse a Renishaw WiRe plain-text export file.

    The first row of the file is skipped (header); the remaining rows are
    expected to contain two tab-separated columns: Raman shift in cm⁻¹ and
    intensity.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the Renishaw ``.txt`` export file.

    Returns
    -------
    dict
        Dictionary with two keys:

        ``"data"`` : pandas.DataFrame
            DataFrame with columns ``"wavenumber"`` (Raman shift in cm⁻¹,
            float) and ``"intensity"`` (float).
        ``"meta"`` : dict
            Provenance metadata containing ``"instrument"``, ``"folder"``,
            and ``"filename"``.
    """
    data = np.loadtxt(file_path, dtype=str, encoding="utf-8", skiprows=1)

    df = pd.DataFrame(data[1:], columns=["wavenumber", "intensity"])
    df["wavenumber"] = df["wavenumber"].astype(float, errors="ignore") # this is actually raman shift in cm-1
    df["intensity"] = df["intensity"].astype(float, errors="ignore")

    meta = {
        "instrument": "Renishaw",
        "folder": str(Path(file_path).parent),
        "filename": Path(file_path).name,
    }
    return {
        "data": df,
        "meta": meta,
    }
