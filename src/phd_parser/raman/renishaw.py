from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, Literal
import pandas as pd
import numpy as np

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def _convert_value(value: str) -> Any:
    try:
        return float(value)
    except ValueError:
        return value

ACCEPTED_FILE_EXTENSIONS = {".txt"}

def read_export(file_path: str | Path) -> Dict[str, Any]:
    """
    Main function to read and parse the Renishaw export file.
    Returns a dictionary with 'meta' and 'data' keys.
    """
    data = np.loadtxt(file_path, dtype=str, encoding="utf-8", skiprows=1)
    
    df = pd.DataFrame(data[1:], columns=["wavelength", "intensity"])
    df["wavelength"] = df["wavelength"].astype(float)
    df["intensity"] = df["intensity"].astype(float)

    meta = {
        "instrument": "Renishaw",
        "folder": str(Path(file_path).parent),
        "filename": Path(file_path).name,
    }
    return {
        "data": df,
        "meta": meta,
    }