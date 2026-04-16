import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Enumerations  (from SpectroChemPy / py-wdf-reader)
# =============================================================================

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
        _map = {
            1: "cm⁻¹", 2: "nm", 3: "nm", 4: "eV", 5: "µm",
            8: "mm", 9: "m", 10: "K", 11: "Pa", 12: "s",
            13: "ms", 19: "°", 20: "rad", 21: "°C",
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


# =============================================================================
# Block-layout constants  (byte offsets within WDF1 header)
# =============================================================================

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


# =============================================================================
# Result dataclass
# =============================================================================

@dataclass
class WDFResult:
    """Container for all data extracted from a .wdf file."""
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

def read_wdf(filename: str | Path) -> WDFResult:
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

# =============================================================================
# Plotting
# =============================================================================

# def plot_wdf(result: WDFResult):
#     """
#     Plot a WDFResult.

#     - Single spectrum   → one line plot
#     - Series            → overlaid spectra
#     - Map               → integrated intensity image  +  mean spectrum
#     """
#     wn  = result.wavenumber
#     dat = result.data
#     lbl = result.x_unit.label()

#     if result.measurement_type == MeasurementType.Mapping and dat.ndim == 3:
#         fig, axes = plt.subplots(1, 2, figsize=(12, 4))
#         im = axes[0].imshow(dat.sum(axis=2), aspect="equal", origin="upper")
#         plt.colorbar(im, ax=axes[0], label="Integrated intensity")
#         axes[0].set_title("Integrated intensity map")
#         axes[0].axis("off")

#         mean_spec = dat.mean(axis=(0, 1))
#         axes[1].plot(wn, mean_spec, linewidth=0.9)
#         axes[1].set_xlabel(f"Raman shift ({lbl})")
#         axes[1].set_ylabel("Intensity")
#         axes[1].set_title("Mean spectrum")
#         axes[1].margins(x=0)

#     elif dat.ndim == 2:
#         fig, ax = plt.subplots(figsize=(9, 4))
#         for row in dat:
#             ax.plot(wn, row, linewidth=0.7, alpha=0.75)
#         ax.set_xlabel(f"Raman shift ({lbl})")
#         ax.set_ylabel("Intensity")
#         ax.set_title(result.title or Path(result.filename).name)
#         ax.margins(x=0)

#     else:
#         fig, ax = plt.subplots(figsize=(9, 4))
#         ax.plot(wn, dat, linewidth=0.9)
#         ax.set_xlabel(f"Raman shift ({lbl})")
#         ax.set_ylabel("Intensity")
#         ax.set_title(result.title or Path(result.filename).name)
#         ax.margins(x=0)

#     fig.suptitle(Path(result.filename).name, fontsize=9, color="gray")
#     plt.tight_layout()
#     plt.show()m


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python wdf_reader.py <file.wdf> [--plot]")
        sys.exit(1)

    r = read_wdf(sys.argv[1], plot="--plot" in sys.argv)

    print(f"File             : {r.filename}")
    print(f"Title            : {r.title}")
    print(f"Measurement type : {r.measurement_type}")
    print(f"Scan type        : {r.scan_type}")
    print(f"Laser            : {r.laser_cm1:.2f} cm⁻¹")
    print(f"Wavenumber range : {r.wavenumber[0]:.1f} – {r.wavenumber[-1]:.1f} cm⁻¹  ({len(r.wavenumber)} pts)")
    print(f"Data shape       : {r.data.shape}")
    print(f"Acq. time        : {r.acq_time}")
    if r.x_pos is not None:
        print(f"X positions      : {r.x_pos.shape}, range {r.x_pos.min():.3f}–{r.x_pos.max():.3f}")
    if r.y_pos is not None:
        print(f"Y positions      : {r.y_pos.shape}, range {r.y_pos.min():.3f}–{r.y_pos.max():.3f}")
    print(f"\nAll meta keys    : {list(r.meta.keys())}")