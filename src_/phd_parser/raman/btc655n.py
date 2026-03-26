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