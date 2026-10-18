"""The tab-separated ``.txt`` export of the Urakawa group's in-house LabVIEW TPR rig.

The rig is a quartz reactor in a furnace with a thermocouple at the bed
(``TC 1``), a TCD on the hydrogen line (``TCD (H2)``), a second TCD
(``TCD (O2)``) and, in the older exports, an FID. LabVIEW writes one row
per second: a short key-value header (user, sample, weight, comment), a
``writing started on d-m-Y at H:M:S`` line, the column line, a blank line
and the data.

Two variants exist and both are read:

* variant 1 (2026-04 to 2026-07): ``time (s)  TCD (H2)  FID  TCD (O2)  TC 1``
* variant 2 (2026-07-23 onwards): ``date  time  time (s)  TCD (H2)  TCD (O2)  TC 1``,
  per-row wall-clock stamps, no FID, and the weight quoted with a decimal
  comma (``"101,9 mg"``)

The number of blank header lines and of trailing tabs varies between files.
Some variant-1 files carry further columns without a name in the column line
(constant numbers, most likely a stale MS block); they are kept under
``column_<n>`` with a warning, as is any named column this module does not
know (see *Instrument exports drift over time* in ``CLAUDE.md``). Detector
signals are stored as the file gives them, in mV and with their raw sign.
"""

import logging
import re
from io import StringIO
from pathlib import Path
from typing import Any, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)

COLUMN_MAP: dict[str, str] = {
    "time (s)": "time_s",
    "TC 1": "temperature_C",
    "TCD (H2)": "tcd_h2",
    "TCD (O2)": "tcd_o2",
    "FID": "fid",
    "date": "date",
    "time": "time",
}
"""Column names of the export mapped onto the stored names.

``time_s`` and ``temperature_C`` are required; ``tcd_h2``, ``tcd_o2`` and
``fid`` are the detector signals; ``date`` and ``time`` (variant 2) are
combined into per-row timestamps. A column that is not listed here is kept
under a sanitised name and logged as a warning.
"""

SIGNAL_NAMES: tuple[str, ...] = ("tcd_h2", "tcd_o2", "fid")
"""Stored names of the detector signals, in mV."""

_START_PATTERN = re.compile(r"writing started on\s+(\d{1,2}-\d{1,2}-\d{4})\s+at\s+(\d{1,2}:\d{2}:\d{2})")
_WEIGHT_PATTERN = re.compile(r"([-+]?\d+(?:[.,]\d+)?)\s*(mg|g)?", re.IGNORECASE)


def _read_text(filepath: Path) -> str:
    raw = filepath.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def _sanitise(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_").lower()
    return cleaned or "column"


def parse_sample_weight_mg(text: Union[str, None]) -> Union[float, None]:
    """Read the sample weight of the header in mg, decimal comma and quotes included.

    Parameters
    ----------
    text : str or None
        The value cell of the ``sample weight`` line, e.g. ``"104.0 mg"`` or
        ``'"101,9 mg"'``.

    Returns
    -------
    float or None
        The weight in mg; a value in g is converted. ``None`` if the cell is
        empty or holds no number.
    """
    if text is None:
        return None
    match = _WEIGHT_PATTERN.search(text.replace('"', "").strip())
    if match is None:
        return None
    value = float(match.group(1).replace(",", "."))
    unit = (match.group(2) or "mg").lower()
    return value * 1000.0 if unit == "g" else value


def parse_start_time(text: str) -> Union[pd.Timestamp, None]:
    """Read ``writing started on d-m-Y at H:M:S`` as a timezone-naive wall-clock timestamp.

    Parameters
    ----------
    text : str
        The header line.

    Returns
    -------
    pandas.Timestamp or None
        The start as the rig's clock showed it (Europe/Amsterdam wall clock,
        no zone attached); ``None`` if the line does not match.
    """
    match = _START_PATTERN.search(text)
    if match is None:
        return None
    return pd.Timestamp(pd.to_datetime(f"{match.group(1)} {match.group(2)}", format="%d-%m-%Y %H:%M:%S"))


def _split_header(lines: list[str]) -> tuple[dict[str, str], Union[pd.Timestamp, None], int]:
    """Return the header key-value pairs, the start time and the index of the column line."""
    column_index = None
    for index, line in enumerate(lines):
        cells = [cell.strip() for cell in line.split("\t")]
        if "time (s)" in cells:
            column_index = index
            break
    if column_index is None:
        raise ValueError("No column line with 'time (s)' found; is this a LabVIEW TPR export?")
    header: dict[str, str] = {}
    start_time = None
    for line in lines[:column_index]:
        cells = line.split("\t")
        key = cells[0].strip()
        if not key:
            continue
        if key.lower().startswith("writing started on"):
            start_time = parse_start_time(key)
            if start_time is None:
                logger.warning("Could not read the start time from %r", key)
            continue
        header[key] = cells[1].strip().strip('"') if len(cells) > 1 else ""
    return header, start_time, column_index


def _read_table(rows: list[str]) -> pd.DataFrame:
    """Parse the data rows; rows with fewer fields are padded with NaN, trailing tabs are tolerated."""
    n_fields = max(row.count("\t") for row in rows) + 1
    return pd.read_csv(StringIO("\n".join(rows)), sep="\t", header=None, names=range(n_fields), engine="c")


def _timestamps(date: pd.Series, time: pd.Series) -> npt.NDArray[np.datetime64]:
    date_text = date.astype("Int64").astype(str).str.replace("<NA>", "", regex=False)
    stamps = pd.to_datetime(date_text + " " + time.astype(str), format="%Y%m%d %H:%M:%S", errors="coerce")
    return stamps.to_numpy(dtype="datetime64[ns]")


def read_export(filepath: Union[str, Path]) -> dict[str, Any]:
    """Parse one LabVIEW TPR export (both column variants).

    Parameters
    ----------
    filepath : str or pathlib.Path
        The ``.txt`` file.

    Returns
    -------
    dict
        ``"data"`` holds ``time_s`` (seconds since the start of writing),
        ``temperature_C`` (``TC 1``), ``signals`` (a dict of the detector
        signals in mV: ``tcd_h2``, ``tcd_o2`` and ``fid`` when the file has
        it), ``extra`` (a dict of every unrecognised or unnamed numeric
        column, by stored name) and, for variant 2, ``timestamp`` (per-row
        wall clock as ``datetime64[ns]``). ``"meta"`` holds ``username``,
        ``sample_name``, ``sample_weight_mg`` (float or None), ``comment``,
        ``start_time`` (timezone-naive :class:`pandas.Timestamp` of the
        rig's wall clock, Europe/Amsterdam; ``None`` if absent),
        ``source_file``, ``columns`` (the original column names, in file
        order), ``column_map`` (original name to stored name for every kept
        column), ``extra_columns`` (stored name to original name or
        1-based column index), ``format_variant`` (``1`` or ``2``),
        ``header`` (every key-value line as text) and ``n_rows``.

    Raises
    ------
    ValueError
        If the column line or the ``time (s)`` / ``TC 1`` columns are
        missing.
    """
    filepath = Path(filepath)
    lines = _read_text(filepath).splitlines()
    header, start_time, column_index = _split_header(lines)
    columns = [cell.strip() for cell in lines[column_index].split("\t")]
    while columns and columns[-1] == "":
        columns.pop()
    rows = [line for line in lines[column_index + 1:] if line.strip()]
    if not rows:
        raise ValueError(f"{filepath.name} holds no data rows")
    table = _read_table(rows)

    stored: dict[str, pd.Series] = {}
    column_map: dict[str, str] = {}
    extra_columns: dict[str, Union[str, int]] = {}
    unknown_named: list[str] = []
    unnamed: list[int] = []
    for position in range(table.shape[1]):
        series = table.iloc[:, position]
        name = columns[position] if position < len(columns) else ""
        if name == "":
            if series.isna().all():
                continue  # trailing tabs
            key = f"column_{position + 1}"
            unnamed.append(position + 1)
            extra_columns[key] = position + 1
        elif name in COLUMN_MAP:
            key = COLUMN_MAP[name]
            column_map[name] = key
        else:
            key = _sanitise(name)
            while key in stored or key in COLUMN_MAP.values():
                key += "_"
            unknown_named.append(name)
            extra_columns[key] = name
            column_map[name] = key
        stored[key] = series
    if unknown_named:
        logger.warning("%s: unrecognised column(s) %s kept as %s", filepath.name, unknown_named,
                       [key for key, original in extra_columns.items() if original in unknown_named])
    if unnamed:
        logger.warning("%s: %d column(s) without a name in the column line (positions %s) kept as column_<n>",
                       filepath.name, len(unnamed), unnamed)
    for required in ("time_s", "temperature_C"):
        if required not in stored:
            original = next(name for name, key in COLUMN_MAP.items() if key == required)
            raise ValueError(f"{filepath.name} has no {original!r} column; columns are {columns}")

    def as_float(key: str) -> npt.NDArray[np.float64]:
        values = pd.to_numeric(stored[key], errors="coerce")
        if values.isna().sum() > stored[key].isna().sum():
            logger.warning("%s: non-numeric entries in %r read as NaN", filepath.name, key)
        return values.to_numpy(dtype=float)

    data: dict[str, Any] = {
        "time_s": as_float("time_s"),
        "temperature_C": as_float("temperature_C"),
        "signals": {name: as_float(name) for name in SIGNAL_NAMES if name in stored},
        "extra": {name: as_float(name) for name in extra_columns},
    }
    format_variant = 2 if "date" in stored and "time" in stored else 1
    if format_variant == 2:
        data["timestamp"] = _timestamps(stored["date"], stored["time"])
        if start_time is None and np.isfinite(data["time_s"][0]) and not np.isnat(data["timestamp"][0]):
            start_time = pd.Timestamp(data["timestamp"][0]) - pd.Timedelta(seconds=float(data["time_s"][0]))
            start_time = start_time.round("s")
    missing = [name for name in SIGNAL_NAMES[:2] if name not in data["signals"]]
    if missing:
        logger.warning("%s: detector column(s) %s not in the file", filepath.name, missing)

    meta: dict[str, Any] = {
        "username": header.get("username"),
        "sample_name": header.get("sample name"),
        "sample_weight_mg": parse_sample_weight_mg(header.get("sample weight")),
        "comment": header.get("comment"),
        "start_time": start_time,
        "source_file": str(filepath),
        "columns": columns,
        "column_map": column_map,
        "extra_columns": extra_columns,
        "format_variant": format_variant,
        "header": header,
        "n_rows": int(table.shape[0]),
    }
    logger.debug("Read %s: variant %d, %d rows, columns %s", filepath.name, format_variant, table.shape[0], columns)
    return {"data": data, "meta": meta}
