import pytest
import numpy as np
import numpy.typing as npt
import os

from phd_parser.xps.casaxps import read_export, extract_lines, split_lines, parse_metadata_lines, parse_data_lines

MOCK_FILE_PATH = os.path.join(os.path.dirname(__file__), "casaxps_export_single.txt")
MOCK_FILE_PATH_DOUBLE = os.path.join(os.path.dirname(__file__), "casaxps_export_double.txt")
KNOWN_METADATA_KEYS = {"info_0", "Characteristic Energy eV", "Acquisition Time s", "Name", "Position", "FWHM", "Area", "Lineshape"}

def test_extract_lines():
    lines = extract_lines(MOCK_FILE_PATH)
    assert len(lines) > 0, "Extracted lines should not be empty"

def test_split_xps_file():
    lines = extract_lines(MOCK_FILE_PATH)
    meta, cols, data = split_lines(lines)

    assert len(meta) > 0, "Metadata lines should not be empty"
    assert cols.startswith("K.E."), "Column header should start with 'K.E.'"
    assert len(data) > 0, "Data lines should not be empty "


def test_parse_metadata():
    lines = extract_lines(MOCK_FILE_PATH)
    meta, _, _ = split_lines(lines)
    metadata = parse_metadata_lines(meta)

    assert isinstance(metadata, dict), "Metadata should be a dictionary"
    assert len(metadata) > 0, "Metadata dictionary should not be empty"
    assert "info_0" in metadata, "Metadata should contain 'info_0' key"
    for key in KNOWN_METADATA_KEYS:
        assert key in metadata, f"Metadata should contain '{key}' key"


def test_parse_data_lines():
    lines = extract_lines(MOCK_FILE_PATH)
    _, cols, data = split_lines(lines)
    ke_df, be_df = parse_data_lines(data, header_line=cols)

    assert not ke_df.empty, "Kinetic energy DataFrame should not be empty"
    assert not be_df.empty, "Binding energy DataFrame should not be empty"
    assert "K.E." in ke_df.columns, "Kinetic energy DataFrame should contain 'K.E.' column"
    assert "B.E." in be_df.columns, "Binding energy DataFrame should contain 'B.E.' column"

def test_read_export():
    xps = read_export(MOCK_FILE_PATH)

    assert "meta" in xps, "Output should contain 'meta' key"
    assert "kinetic energy" in xps, "Output should contain 'kinetic energy' key"
    assert "binding energy" in xps, "Output should contain 'binding energy' key"

    metadata = xps["meta"]
    ke_df = xps["kinetic energy"]
    be_df = xps["binding energy"]

    assert isinstance(metadata, dict), "Metadata should be a dictionary"
    assert len(metadata) > 0, "Metadata should not be empty"
    for key in KNOWN_METADATA_KEYS:
        assert key in metadata, f"Metadata should contain '{key}' key, but it is missing"
    assert not ke_df.empty, "Kinetic energy DataFrame should not be empty"
    assert not be_df.empty, "Binding energy DataFrame should not be empty"