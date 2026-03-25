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


def extract_lines(file_path: str | Path) -> List[str]:
    """
    Read file and return raw lines.
    """
    path = Path(file_path)

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        logger.info(f"Extracted {len(lines)} lines from {file_path}")
        return lines