from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, Literal
import pandas as pd

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def _convert_value(value: str) -> Any:
    try:
        return float(value)
    except ValueError:
        return value


ACCEPTED_FILE_EXTENSIONS = {".txt"}
COLUME_LINE_START_TOKEN = "Pixel"

X_KEYS = Literal["Pixel", "Wavelength", "Wavenumber", "Raman Shift"]
Y_KEYS = Literal["Dark", "Reference", "Raw data #1", "Dark Subtracted #1", "%TR #1", "Absorbance #1", "Irradiance (lumen) #1"]

def extract_lines(file_path: str | Path) -> List[str]:
    """Read a file and return its raw lines as a list of strings.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the file to read.

    Returns
    -------
    list of str
        All lines of the file, including newline characters.
    """
    path = Path(file_path)

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        logger.info(f"Extracted {len(lines)} lines from {file_path}")
        return lines


def split_lines(
    lines: List[str],
) -> Tuple[List[str], str, List[str]]:
    """Split raw file lines into metadata lines, a column header, and data lines.

    Scans ``lines`` in order until a line whose first tab-separated token
    starts with ``"Pixel"`` is found.  Everything before that line is
    considered metadata; everything after is considered data.

    Parameters
    ----------
    lines : list of str
        Raw lines as returned by :func:`extract_lines`.

    Returns
    -------
    tuple of (list of str, str, list of str)
        A 3-tuple ``(metadata_lines, column_line, data_lines)`` where
        ``metadata_lines`` contains all lines before the column header,
        ``column_line`` is the header row itself, and ``data_lines`` contains
        all non-empty lines after the header.

    Raises
    ------
    ValueError
        When no column header row starting with ``"Pixel"`` is found.
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

        if first_token.startswith(COLUME_LINE_START_TOKEN):
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
    """Parse semicolon-delimited metadata lines into a key-value dictionary.

    Each non-empty line is split on ``";"`` and the first two tokens are
    used as key and value respectively.  Values that can be interpreted as
    floats are converted; all others are kept as strings.  Duplicate keys
    are overwritten by the last occurrence.

    Parameters
    ----------
    lines : list of str
        Metadata lines as returned by the first element of
        :func:`split_lines`.

    Returns
    -------
    dict
        Mapping of metadata key strings to parsed values (``float`` or
        ``str``).
    """
    metadata: Dict[str, Any] = {}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        parts = [p.strip() for p in stripped.split(";")]

        # require key + value
        if len(parts) < 2:
            continue

        key = parts[0]
        value = _convert_value(parts[1])

        metadata[key] = value  # overwrite if duplicate

    return metadata

def parse_data_lines(
    data_lines: List[str],
    header_line: Optional[str] = None,
    remove_empty: bool = True,
) -> pd.DataFrame:
    """Parse semicolon-delimited data lines into a DataFrame.

    Rows are split on ``";"`` and the trailing empty column produced by a
    trailing delimiter is dropped.  Numeric conversion is applied to all
    columns via ``pd.to_numeric``.

    Parameters
    ----------
    data_lines : list of str
        Data lines as returned by the third element of :func:`split_lines`.
    header_line : str, optional
        The column header line used to name DataFrame columns.  When
        ``None``, columns are named ``col_0``, ``col_1``, …
    remove_empty : bool, optional
        Drop rows where the ``"Wavelength"`` column is ``NaN`` and reset the
        index when ``True`` (default).

    Returns
    -------
    pandas.DataFrame
        Parsed data with one column per field and numeric dtypes where
        possible.
    """
    # --- header ---
    if header_line is not None:
        columns = [c.strip() for c in header_line.strip().split(";") if c.strip()]
    else:
        columns = [
            f"col_{i}"
            for i in range(len(data_lines[0].strip().split(";")) - 1)
        ]

    # --- rows ---
    rows = [
        [cell.strip() if cell.strip() else None for cell in line.strip().split(";")][:-1] # drop last empty column
        for line in data_lines
        if line.strip()
    ]
    print(rows[:5])

    df = pd.DataFrame(rows, columns=columns)

    # --- numeric conversion ---
    df = df.apply(pd.to_numeric, errors="coerce")

    if remove_empty:
        df.dropna(how="all", subset=["Wavelength"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df

def read_export(file_path: str | Path, remove_empty: bool = True) -> Dict[str, Any]:
    """Read and parse a B&W Tek BTC655N spectrometer export file.

    Only ``.txt`` files are supported.  The function delegates line
    extraction, splitting, metadata parsing, and data parsing to the
    lower-level helpers in this module.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the BTC655N ``.txt`` export file.
    remove_empty : bool, optional
        Passed through to :func:`parse_data_lines`; drops rows with no
        wavelength data when ``True`` (default).

    Returns
    -------
    dict
        Dictionary with two keys:

        ``"meta"`` : dict
            Provenance metadata parsed from the file header, with
            ``"filename"`` added automatically.
        ``"data"`` : pandas.DataFrame
            Parsed spectral data with one column per field (e.g.
            ``"Pixel"``, ``"Wavelength"``, ``"Raw data #1"``).

    Raises
    ------
    ValueError
        When the file extension is not in ``{".txt"}``.
    """
    path = Path(file_path)


    if path.suffix.lower() not in ACCEPTED_FILE_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    # --- read ---
    lines = extract_lines(path)
    metadata_lines, header_line, data_lines = split_lines(lines)

    # --- parse ---
    metadata = parse_metadata_lines(metadata_lines)
    metadata["filename"] = path.name
    parsed = parse_data_lines(data_lines, header_line, remove_empty=remove_empty)

    return {
        "meta": metadata,
        "data": parsed,
    }
