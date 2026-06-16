from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _convert_value(value: str) -> Any:
    """Convert string to int/float if possible, otherwise return string."""
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value

ACCEPTED_FILE_EXTENSIONS = {".txt", ".csv"}

def extract_lines(file_path: str | Path) -> List[str]:
    """Read a CasaXPS export file and return its raw lines.

    Parameters
    ----------
    file_path : str or Path
        Path to the CasaXPS export file (``.txt`` or ``.csv``).

    Returns
    -------
    list of str
        All lines of the file including newline characters.
    """
    path = Path(file_path)

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        logger.info(f"Extracted {len(lines)} lines from {file_path}")
        return lines

def split_lines(
    lines: List[str],
) -> Tuple[List[str], str, List[str]]:
    """Partition raw file lines into metadata, column-header, and data sections.

    The column-header row is identified as the first non-empty line
    whose leading token starts with ``'K.E.'``.  All lines before it
    are treated as metadata; all non-empty lines after it are data.

    Parameters
    ----------
    lines : list of str
        Raw lines as returned by :func:`extract_lines`.

    Returns
    -------
    metadata_lines : list of str
        Lines preceding the column-header row.
    column_line : str
        The column-header row (tab-separated column names).
    data_lines : list of str
        Non-empty lines following the column-header row.

    Raises
    ------
    ValueError
        When no column header row starting with ``'K.E.'`` is found.
    """
    metadata_lines: List[str] = []
    column_line: str | None = None
    data_start_idx: int | None = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            metadata_lines.append(line)
            continue

        first_token = stripped.split("\t")[0].strip()

        if first_token.startswith("K.E."):
            column_line = line
            data_start_idx = i + 1
            break

        metadata_lines.append(line)

    if column_line is None or data_start_idx is None:
        raise ValueError("Could not find column header row starting with 'K.E.'")

    data_lines = [
        line for line in lines[data_start_idx:] if line.strip()
    ]

    logger.debug(f"Split lines into {len(metadata_lines)} metadata lines, 1 column header line, and {len(data_lines)} data lines")

    return metadata_lines, column_line, data_lines

def parse_metadata_lines(lines: List[str]) -> Dict[str, Any]:
    """Parse CasaXPS metadata lines into a structured dictionary.

    Tab-separated lines are parsed as key-value pairs.  Lines with
    four or more tokens are treated as alternating key-value sequences.
    Single-token lines are stored under auto-incremented ``info_N``
    keys.  When a ``"Name"`` key is present the function also builds a
    ``"peaks"`` list and assigns a ``"type"`` label
    (``"singlet"``, ``"doublet"``, ``"triplet"``, or ``"unknown"``).

    Parameters
    ----------
    lines : list of str
        Metadata lines as returned by :func:`split_lines`.

    Returns
    -------
    dict
        Parsed metadata including an optional ``"peaks"`` list and a
        ``"type"`` string.
    """
    metadata: Dict[str, Any] = {}
    info_counter = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        parts = [p.strip() for p in stripped.split("\t") if p.strip()]

        # --- free text ---
        if len(parts) == 1:
            metadata[f"info_{info_counter}"] = parts[0]
            info_counter += 1
            continue

        # --- detect alternating key-value pairs (multi-key line) ---
        if len(parts) > 2 and all(i % 2 == 0 for i in range(len(parts))):
            # fallback safety (unlikely case)
            pass

        # better detection: alternating key-value pattern
        if len(parts) >= 4:
            is_alternating = True
            for i in range(1, len(parts), 2):
                # values should be convertible OR strings
                _ = _convert_value(parts[i])
            if is_alternating:
                for i in range(0, len(parts) - 1, 2):
                    key = parts[i]
                    value = _convert_value(parts[i + 1])
                    metadata.setdefault(key, []).append(value)
                continue

        # --- normal key + values ---
        key = parts[0]
        values = [_convert_value(v) for v in parts[1:]]

        metadata.setdefault(key, []).extend(values)

    # --- determine peak count from Name ---
    peak_count = len(metadata["Name"]) if "Name" in metadata else None

    # --- build structured peaks ---
    peaks = []
    if peak_count:
        for i in range(peak_count):
            peak = {}
            for key, values in metadata.items():
                if isinstance(values, list) and len(values) == peak_count:
                    peak[key] = values[i]
            peaks.append(peak)

    if peaks:
        metadata["peaks"] = peaks

    # --- assign type ---
    if peak_count == 1:
        metadata["type"] = "singlet"
    elif peak_count == 2:
        metadata["type"] = "doublet"
    elif peak_count == 3:
        metadata["type"] = "triplet"
    else:
        metadata["type"] = "unknown"

    return metadata


def parse_data_lines(
    data_lines: List[str],
    header_line: Optional[str] = None,
) -> pd.DataFrame |  Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse CasaXPS data lines into one or two DataFrames.

    When the column-header row contains an empty tab-separated token
    the data are split at that position into a kinetic-energy (K.E.)
    DataFrame and a binding-energy (B.E.) DataFrame.  Otherwise a
    single DataFrame is returned.

    Parameters
    ----------
    data_lines : list of str
        Tab-separated numeric data lines.
    header_line : str, optional
        The column-header row.  When ``None`` column names are
        auto-generated as ``col_0``, ``col_1``, …

    Returns
    -------
    pandas.DataFrame or tuple of (pandas.DataFrame, pandas.DataFrame)
        A single numeric DataFrame when no separator column is detected,
        or a ``(left, right)`` tuple when a blank separator column is
        found.
    """
    # --- parse header ---
    if header_line is not None:
        columns = [c.strip() for c in header_line.strip().split("\t")]
    else:
        # fallback: infer column count from first row
        columns = [
            f"col_{i}"
            for i in range(len(data_lines[0].strip().split("\t")))
        ]

    # --- parse rows ---
    rows: List[List[str]] = [
        [cell.strip() for cell in line.strip().split("\t")]
        for line in data_lines
        if line.strip()
    ]

    # --- detect separator column ---
    split_idx = None
    for i, col in enumerate(columns):
        if col == "":
            split_idx = i
            break

    # --- no split → single dataframe ---
    if split_idx is None:
        df = pd.DataFrame(rows, columns=columns)
        return df.apply(pd.to_numeric, errors="coerce")

    # --- split into two datasets ---
    left_cols = columns[:split_idx]
    right_cols = columns[split_idx + 1 :]

    left_data = [row[:split_idx] for row in rows]
    right_data = [row[split_idx + 1 :] for row in rows]

    df_left = pd.DataFrame(left_data, columns=left_cols)
    df_right = pd.DataFrame(right_data, columns=right_cols)

    # --- numeric conversion ---
    df_left = df_left.apply(pd.to_numeric, errors="coerce")
    df_right = df_right.apply(pd.to_numeric, errors="coerce")

    return df_left, df_right


def read_export(file_path: str | Path) -> Dict[str, Any]:
    """Parse a CasaXPS export file into metadata and spectral DataFrames.

    Supports ``.txt`` and ``.csv`` file extensions.  The file is split
    into metadata and data sections by :func:`split_lines`, then each
    section is parsed independently.  Kinetic-energy and binding-energy
    datasets are identified by the name of their first column.

    Parameters
    ----------
    file_path : str or Path
        Path to the CasaXPS export file.

    Returns
    -------
    dict
        Dictionary with keys:

        ``"meta"`` : dict
            Parsed metadata including peak information and type label.
        ``"kinetic energy"`` : pandas.DataFrame or None
            K.E.-axis spectral data, or ``None`` when absent.
        ``"binding energy"`` : pandas.DataFrame or None
            B.E.-axis spectral data, or ``None`` when absent.

    Raises
    ------
    ValueError
        When the file extension is not ``.txt`` or ``.csv``, or when
        no ``'K.E.'`` column header row is found.
    """
    path = Path(file_path)

    # --- validate file ---
    if path.suffix.lower() not in ACCEPTED_FILE_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    # --- read ---
    lines = extract_lines(path)

    # --- split ---
    metadata_lines, header_line, data_lines = split_lines(lines)

    # --- parse metadata ---
    metadata = parse_metadata_lines(metadata_lines)
    metadata["filename"] = path.name

    # --- parse data ---
    parsed = parse_data_lines(data_lines, header_line)

    # --- normalize output ---
    ke_df: Optional[pd.DataFrame] = None
    be_df: Optional[pd.DataFrame] = None

    if isinstance(parsed, tuple):
        left, right = parsed

        # assign based on first column name (robust, not positional assumption)
        if left.columns[0].startswith("K.E"):
            ke_df, be_df = left, right
        else:
            ke_df, be_df = right, left
    else:
        df = parsed

        # single dataset → detect type
        first_col = df.columns[0]

        if str(first_col).startswith("K.E"):
            ke_df = df
        elif str(first_col).startswith("B.E"):
            be_df = df
        else:
            # unknown → treat as KE by default (raw measurement)
            ke_df = df

    return {
        "meta": metadata,
        "kinetic energy": ke_df,
        "binding energy": be_df,
    }
