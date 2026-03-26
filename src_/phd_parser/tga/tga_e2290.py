import pandas as pd
from io import StringIO
from typing import List, Tuple
import logging

from phd_parser.tga.tga import TGA
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_tga_e2290(data: dict, baseline: dict) -> TGA:
    tga = TGA(
        temperature=data["data"]["Ts"].values,
        mass=data["data"]["Value"].values
    )
    return tga

# --------------------------------------
# Process Header, Data, Footer
# --------------------------------------

def read_tga_e2290(filepath: str):
    """
    Extracts header, data, and footer sections from the txt file.
    """
    if not filepath.endswith(".txt"):
        raise ValueError("File must be a .txt file")

    with open(filepath, "r") as f:
        lines = f.readlines()

    # Find section markers
    header_start = 0
    data_start = 0
    results_start = 0
    sample_start = 0
    method_start = 0

    for i, line in enumerate(lines):
        if line.strip().startswith("Curve Name:"):
            header_start = i
        elif line.strip().startswith("Curve Values:"):
            data_start = i
        elif line.strip().startswith("Results:"):
            results_start = i
        elif line.strip().startswith("Sample:"):
            sample_start = i
        elif line.strip().startswith("Method:"):
            method_start = i
            break

    if 0 in (header_start, data_start, results_start, sample_start, method_start):
        raise ValueError("File does not match expected format (missing section markers).")

    res = {}
    # Extract header (everything between "Curve Name:" and "Curve Values:")
    header_lines = lines[header_start:data_start]
    res["curve_name"], res["saved_at"], res["performed_at"] = parse_header(header_lines)

    # Extract data (everything between "Curve Values:" and "Results:")
    data_lines = lines[data_start+1:results_start]  # +1 to skip "Curve Values:"
    res["data"], res["units"] = parse_data(data_lines)

    # Extract results (from "Results:" to "Sample:")
    results_lines = lines[results_start+1:sample_start]
    results_text = "".join(results_lines).strip()
    res["results"] = results_text

    sample_lines = lines[sample_start+1:method_start]
    res["sample_name"], res["weight"] = parse_sample(sample_lines)

    # Extract footer (from "Results:" to the end)
    method_lines = lines[method_start+1:method_start+2]
    method_text = "".join(method_lines).strip()
    res["method"] = method_text

    return res


# --------------------------------------
# Process Header, Data, Footer
# --------------------------------------
def parse_data(lines: List[str]) -> Tuple[pd.DataFrame, List[str]]:
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
    return pd.read_csv(StringIO("".join(lines[2:])), sep="\s+", names=columns), units


def parse_header(lines: List[str]) -> Tuple[str, pd.Timestamp, pd.Timestamp]:
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

def parse_sample(lines: List[str]) -> Tuple[str | None, float | None]:
    sample_name = None
    weight = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        sample_name = ",".join(line.split(",")[:-1]).strip()
        weight = float(line.split(",")[-1].replace("mg", "").strip())
    return sample_name, weight