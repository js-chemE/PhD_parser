from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
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


def extract_lines(file_path: str | Path) -> List[str]:
    """
    Read file and return raw lines.
    """
    path = Path(file_path)

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        logger.info(f"Extracted {len(lines)} lines from {file_path}")
        return lines
    

def split_lines(
    lines: List[str],
) -> Tuple[List[str], str, List[str]]:
    
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
) -> pd.DataFrame:

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
        [cell.strip() for cell in line.strip().split(";") if cell.strip()]
        for line in data_lines
        if line.strip()
    ]

    df = pd.DataFrame(rows, columns=columns)

    # --- numeric conversion ---
    df = df.apply(pd.to_numeric, errors="coerce")

    return df

def read_export(file_path: str | Path) -> Dict[str, Any]:

    path = Path(file_path)

    
    if path.suffix.lower() not in ACCEPTED_FILE_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    # --- read ---
    lines = extract_lines(path)
    metadata_lines, header_line, data_lines = split_lines(lines)

    # --- parse ---
    metadata = parse_metadata_lines(metadata_lines)
    metadata["filename"] = path.name
    parsed = parse_data_lines(data_lines, header_line)

    return {
        "meta": metadata,
        "data": parsed,
    }