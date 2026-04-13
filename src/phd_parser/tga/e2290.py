import pandas as pd
from io import StringIO
from typing import List, Tuple
import logging
from pathlib import Path

from phd_parser.tga.core import TGAData
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# --------------------------------------
# Read TGA E2290
# --------------------------------------

def read_export(filepath: str | Path):
    """
    Extracts header, data, and footer sections from the txt file.
    """
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
        
    if not filepath.suffix == ".txt":
        raise ValueError("File must be a .txt file")

    lines = extract_lines(filepath)

    header_lines, data_lines, results_lines, sample_lines, method_lines = split_sections(lines)

    res = {}
    res["curve_name"], res["saved_at"], res["performed_at"] = parse_header_lines(header_lines)
    res["data"], res["units"] = parse_data_lines(data_lines)
    res["results"] = "".join(results_lines).strip()
    res["sample_name"], res["weight"] = parse_sample_lines(sample_lines)
    res["method"] = "".join(method_lines).strip()

    return res


# --------------------------------------
# Process Header, Data, Footer
# --------------------------------------
def extract_lines(filepath: str | Path) -> List[str]:
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
        
    with open(filepath, "r") as f:
        lines = f.readlines()
    return lines

def split_sections(lines: List[str]) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    header_lines = []
    data_lines = []
    results_lines = []
    sample_lines = []
    method_lines = []

    current_section = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Curve Name:"):
            current_section = "header"
        elif stripped.startswith("Curve Values:"):
            current_section = "data"
        elif stripped.startswith("Results:"):
            current_section = "results"
        elif stripped.startswith("Sample:"):
            current_section = "sample"
        elif stripped.startswith("Method:"):
            current_section = "method"

        if current_section == "header":
            header_lines.append(line)
        elif current_section == "data":
            data_lines.append(line)
        elif current_section == "results":
            results_lines.append(line)
        elif current_section == "sample":
            sample_lines.append(line)
        elif current_section == "method":
            method_lines.append(line)

    return header_lines[1:], data_lines[1:], results_lines[1:], sample_lines[1:], method_lines[1:]

def parse_data_lines(lines: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extracts data from the data section.
    """
    for line in lines:
        line = line.strip()
        if not line:
            continue
    columns = re.split(r"\s+", lines[0].strip())
    units = re.split(r"\s+", lines[1].strip())
    units = [u.replace("[", "").replace("]", "") for u in units if u]  # Remove empty strings
    df = pd.read_csv(StringIO("".join(lines[2:])), sep="\s+", names=columns)
    return df, units


def parse_header_lines(lines: List[str]) -> Tuple[str, pd.Timestamp, pd.Timestamp]:
    curve_name = None
    performed_at = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("Curve Name:"):
            continue
        elif line.startswith("Performed"):
            performed_at = pd.Timestamp(line.replace("Performed", "").strip())
        else:
            saved_at = pd.Timestamp(line.split(",")[-1].strip())
            curve_name = ",".join(line.split(",")[:-1])
    return curve_name, saved_at, performed_at # type: ignore

def parse_sample_lines(lines: List[str]) -> Tuple[str | None, float | None]:
    sample_name = None
    weight = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        sample_name = ",".join(line.split(",")[:-1]).strip()
        weight = float(line.split(",")[-1].replace("mg", "").strip())
    return sample_name, weight