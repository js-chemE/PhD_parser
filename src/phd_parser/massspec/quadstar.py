from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd

import logging

logger = logging.getLogger(__name__)

def _convert_value(value: str) -> Any:
    """Convert string to int/float if possible, otherwise return string."""
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value

ACCEPTED_FILE_EXTENSIONS = {".asc"}

def extract_lines(file_path: str | Path) -> List[str]:
    """Read a Quadstar ``.asc`` file and return its raw lines.

    Parameters
    ----------
    file_path : str or Path
        Path to the ``.asc`` file to read.

    Returns
    -------
    list[str]
        All lines from the file, including newline characters.
    """
    path = Path(file_path)

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        logger.info(f"Extracted {len(lines)} lines from {file_path}")
        return lines


def split_lines(
    lines: List[str],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Partition raw file lines into the four structural sections of an ``.asc`` export.

    The Quadstar ASCII format uses blank lines as section delimiters.  The
    function collects contiguous non-blank groups and assigns them to the four
    expected sections: file header, cycle/block counts, block definitions, and
    data rows (column header + measurement rows).

    Parameters
    ----------
    lines : list[str]
        Raw lines as returned by :func:`extract_lines`.

    Returns
    -------
    meta_0 : list[str]
        File-header section (ASCII sample cycles, date/time, converted cycles).
    meta_1 : list[str]
        Cycle and datablock count section.
    meta_blocks : list[str]
        Datablock and channel definition section.
    data : list[str]
        Flattened list of column-header and data rows.

    Raises
    ------
    ValueError
        If fewer than four blank-line-delimited sections are found.
    """
    # Collect contiguous non-empty groups separated by blank lines
    sections: List[List[str]] = []
    current: List[str] = []

    for line in lines:
        if line.strip():
            current.append(line)
        else:
            if current:
                sections.append(current)
                current = []

    if current:
        sections.append(current)

    if len(sections) < 4:
        raise ValueError(
            f"Expected at least 4 blank-line-delimited sections, got {len(sections)}"
        )

    meta_0, meta_1, meta_blocks, *rest = sections

    # Flatten remaining sections (column header + data rows) and strip blanks
    data = [line for section in rest for line in section if line.strip()]

    logger.debug(
        f"Split into meta_0={len(meta_0)}, meta_1={len(meta_1)}, "
        f"meta_blocks={len(meta_blocks)}, data={len(data)} lines"
    )

    return meta_0, meta_1, meta_blocks, data


def parse_metadata_lines(
    meta_0: List[str],
    meta_1: List[str],
    meta_blocks: List[str],
) -> Dict[str, Any]:
    """Parse the three metadata sections of a Quadstar ``.asc`` export into a dict.

    Parameters
    ----------
    meta_0 : list[str]
        File-header section lines (file name, date, time, converted cycles).
    meta_1 : list[str]
        Cycle and datablock count section lines.
    meta_blocks : list[str]
        Datablock and channel definition section lines.

    Returns
    -------
    dict[str, Any]
        Flat metadata dictionary containing keys such as ``"file_name"``,
        ``"date"``, ``"time"``, ``"n_cycles"``, ``"n_datablocks"``, and
        ``"datablocks"`` (a nested dict keyed by block ID).
    """
    meta: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # meta_0  –  file header
    # ------------------------------------------------------------------
    def _parse_kv_line(line: str) -> Tuple[str, str]:
        """Split 'KEY : value' on the first ' : ' separator."""
        key, _, value = line.partition(" :")
        return key.strip(), value.strip()

    KEY_MAP_0 = {
        "ASCII SAMPLE CYCLES": "file_name",
        "DATE":                 None,   # handled specially (DATE + TIME on same line)
        "CONVERTED CYCLES":     "converted_cycles",
    }

    for line in meta_0:
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split("\t")
        first = tokens[0].strip()

        if first.startswith("ASCII SAMPLE CYCLES"):
            meta["file_name"] = tokens[-1].strip() if len(tokens) > 1 else ""

        elif first.startswith("DATE"):
            # DATE :	26-3-2026	TIME :	14:57:59
            try:
                meta["date"] = tokens[1].strip()
                meta["time"] = tokens[3].strip()
            except IndexError:
                logger.warning(f"Could not fully parse DATE/TIME line: {line!r}")

        elif first.startswith("CONVERTED CYCLES"):
            meta["converted_cycles"] = _convert_value(tokens[-1].strip())

    # ------------------------------------------------------------------
    # meta_1  –  cycle / datablock counts
    # ------------------------------------------------------------------
    KEY_MAP_1 = {
        "Number of stored cycles":    "n_cycles",
        "Printed start cycle":        "start_cycle",
        "Printed end cycle":          "end_cycle",
        "Number of stored datablocks":"n_datablocks",
    }

    for line in meta_1:
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split("\t")
        first = tokens[0].strip()
        for src_key, dict_key in KEY_MAP_1.items():
            if first.startswith(src_key):
                meta[dict_key] = _convert_value(tokens[-1].strip())
                break

    # ------------------------------------------------------------------
    # meta_blocks  –  datablock + channel definitions
    # ------------------------------------------------------------------
    datablocks: Dict[int, Any] = {}
    current_block_id: int | None = None

    for line in meta_blocks:
        stripped = line.strip()
        if not stripped:
            continue
        tokens = [t.strip() for t in stripped.split("\t")]
        first = tokens[0]

        if first.startswith("Datablock"):
            # e.g.  "Datablock 0\tIon Current\t[A]"
            current_block_id = int(first.split()[1])
            block_type = tokens[1] if len(tokens) > 1 else ""
            unit_raw   = tokens[2] if len(tokens) > 2 else ""
            unit       = unit_raw.strip("[]")
            datablocks[current_block_id] = {
                "type":     block_type,
                "unit":     unit,
                "channels": {},
            }

        elif first.startswith("'") and current_block_id is not None:
            # e.g.  "'0/0'\t2.00\tmin:\t5.12E-13\tmax:\t4.05E-10\t..."
            channel_id = first.strip("'")
            channel: Dict[str, Any] = {}

            # mass: second token (numeric), except PKR blocks which use a label
            mass_token = tokens[1] if len(tokens) > 1 else ""
            channel["mass"] = _convert_value(mass_token)

            # parse remaining key: value pairs  (min:, max:, T_min:, T_max:, ...)
            i = 2
            while i < len(tokens) - 1:
                k = tokens[i].rstrip(":")
                v = tokens[i + 1]
                # skip qualifier tokens that are not key: value pairs
                if k and not v.endswith(":"):
                    channel[k] = _convert_value(v)
                    i += 2
                else:
                    i += 1

            datablocks[current_block_id]["channels"][channel_id] = channel

    meta["datablocks"] = datablocks

    logger.debug(
        f"Parsed metadata: {len(datablocks)} datablock(s), keys={list(meta.keys())}"
    )

    return meta

def parse_data_lines(data: List[str], drop_threshold_cols: bool = False, tz_str: str = "Europe/Amsterdam") -> pd.DataFrame:
    """Parse tab-separated data rows from a Quadstar ``.asc`` export into a DataFrame.

    The first element of ``data`` is treated as the column-header row.
    Numeric columns are coerced to float/int where possible.  A combined
    ``Timestamp`` column is synthesised from the ``Date`` and ``Time`` columns
    and localised to the specified timezone.

    Parameters
    ----------
    data : list[str]
        Raw data lines with the column header as the first element, as
        returned by :func:`split_lines`.
    drop_threshold_cols : bool, optional
        If ``True``, columns whose names start with ``"Threshold"`` are
        removed from the result.  Default is ``False``.
    tz_str : str, optional
        Timezone name used to localise parsed timestamps.  Default is
        ``"Europe/Amsterdam"``.

    Returns
    -------
    pandas.DataFrame
        Parsed measurement table with a ``Timestamp`` column appended.

    Raises
    ------
    ValueError
        If ``data`` is empty.
    """
    if not data:
        logger.error("No data lines to parse")
        raise ValueError("data is empty")

    header = data[0].strip().split("\t")
    rows = [line.strip().split("\t") for line in data[1:] if line.strip()]

    df = pd.DataFrame(rows, columns=header)
    df = df.apply(lambda col: pd.to_numeric(col, errors="coerce").fillna(col))

    # Timestamp
    df["Timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d-%m-%Y %H:%M:%S:%f",
        utc = False,
        errors = "coerce",
    )
    df["Timestamp"] = df["Timestamp"].dt.tz_localize(tz_str, ambiguous="NaT")


    # Threshold columns to category

    if drop_threshold_cols:
        threshold_cols = [col for col in df.columns if col.startswith("Threshold")]
        df.drop(columns=threshold_cols, inplace=True)
    else:
        for i, col in enumerate(df.columns):
            if col == "Threshold":
                df.iloc[:, i] = df.iloc[:, i].astype("category")


    logger.debug(f"Parsed data into DataFrame with shape {df.shape}")

    return df


def _build_column_map(meta: Dict[str, Any], current_columns: List[str]) -> Dict[str, Any]:
    # Flat lookup: channel_id -> (new_name, unit, block_id)
    channel_lookup: Dict[str, Tuple[str, str, str]] = {}
    for block_id, block in meta["datablocks"].items():
        unit = block["unit"]
        for channel_id, channel in block["channels"].items():
            mass = channel["mass"]
            new_name = f"m{mass}" if isinstance(mass, (int, float)) else str(mass)
            channel_lookup[f"'{channel_id}'"] = (new_name, unit, str(block_id))

    meta_columns = {"Cycle", "Date", "Time", "Timestamp"}
    unit_map = {"RelTime[s]": "s"}

    new_column_names, units, sources = [], [], []
    threshold_counters: Dict[str, int] = {}
    current_block: str = "meta"

    for col in current_columns:
        if col in meta_columns:
            new_column_names.append(col)
            units.append("")
            sources.append("meta")

        elif col in unit_map:
            new_column_names.append(col)
            units.append(unit_map[col])
            sources.append("meta")

        elif col in channel_lookup:
            new_name, unit, block_id = channel_lookup[col]
            current_block = block_id
            new_column_names.append(new_name)
            units.append(unit)
            sources.append(block_id)

        elif col == "Threshold":
            count = threshold_counters.get(current_block, 0)
            threshold_counters[current_block] = count + 1
            new_column_names.append(f"Thresh    old_{current_block}_{count}")
            units.append("")
            sources.append(current_block)

        else:
            new_column_names.append(col)
            units.append("")
            sources.append("meta")

    return {
        "original": current_columns,
        "new": new_column_names,
        "units": units,
        "source": sources,
    }


def read_export_single(file_path: str | Path, drop_threshold_cols: bool = True, tz_str: str = "Europe/Amsterdam") -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Parse a single Quadstar ``.asc`` file into metadata and a DataFrame.

    Orchestrates :func:`extract_lines`, :func:`split_lines`,
    :func:`parse_metadata_lines`, and :func:`parse_data_lines`, then
    renames DataFrame columns according to the datablock/channel definitions
    found in the metadata and attaches a ``"column_map"`` key to the returned
    metadata dict.

    Parameters
    ----------
    file_path : str or Path
        Path to the ``.asc`` file.
    drop_threshold_cols : bool, optional
        Passed through to :func:`parse_data_lines`.  Drops threshold columns
        when ``True`` (default).
    tz_str : str, optional
        Timezone name for timestamp localisation (default ``"Europe/Amsterdam"``).

    Returns
    -------
    meta : dict[str, Any]
        Parsed file metadata including ``"datablocks"`` and ``"column_map"``.
    df : pandas.DataFrame
        Measurement table with renamed columns.
    """
    lines = extract_lines(file_path)
    meta_0, meta_1, meta_blocks, data = split_lines(lines)
    meta = parse_metadata_lines(meta_0, meta_1, meta_blocks)
    df = parse_data_lines(data, drop_threshold_cols=drop_threshold_cols, tz_str=tz_str)

    column_map = _build_column_map(meta, list(df.columns))
    df = df.rename(columns=dict(zip(
        column_map["original"],
        column_map["new"],
    )))

    meta["column_map"] = column_map

    logger.info(f"Read export: {df.shape[0]} cycles, {df.shape[1]} columns")

    return meta, df

def read_export(path: str | Path, drop_threshold_cols: bool = True, tz_str: str = "Europe/Amsterdam") -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Load one or more Quadstar ``.asc`` exports from a file or directory.

    When ``path`` points to a single file, delegates directly to
    :func:`read_export_single`.  When ``path`` is a directory, all ``.asc``
    files found at the top level are parsed and their DataFrames are
    concatenated row-wise with a reset index.

    Parameters
    ----------
    path : str or Path
        Path to a single ``.asc`` file or a directory containing ``.asc``
        files.
    drop_threshold_cols : bool, optional
        Passed through to :func:`read_export_single`.  Default is ``True``.
    tz_str : str, optional
        Timezone name for timestamp localisation (default ``"Europe/Amsterdam"``).

    Returns
    -------
    meta : dict[str, Any]
        For a single file: the metadata dict from :func:`read_export_single`.
        For a directory: a dict with ``"files_processed"`` (int) and
        ``"file_metadata"`` (list of per-file metadata dicts).
    df : pandas.DataFrame
        Combined measurement table.

    Raises
    ------
    ValueError
        If ``path`` is a directory but contains no ``.asc`` files.
    """
    path = Path(path)

    if path.is_file():
        return read_export_single(path, drop_threshold_cols=drop_threshold_cols, tz_str=tz_str)

    elif path.is_dir():
        all_meta = []
        all_dfs = []
        for file in path.iterdir():
            if file.suffix.lower() in ACCEPTED_FILE_EXTENSIONS:
                logger.info(f"Processing file: {file.name}")
                meta, df = read_export_single(file, drop_threshold_cols=drop_threshold_cols, tz_str=tz_str)
                all_meta.append(meta)
                all_dfs.append(df)
            else:
                logger.warning(f"Skipping unsupported file: {file.name}")

        if not all_dfs:
            raise ValueError(f"No valid data files found in directory: {path}")

        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Combined DataFrame shape from directory: {combined_df.shape}")
        return {"files_processed": len(all_dfs), "file_metadata": all_meta}, combined_df
