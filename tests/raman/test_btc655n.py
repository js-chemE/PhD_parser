import numpy.typing as npt
import os

from phd_parser.raman.btc655n import extract_lines, split_lines, parse_metadata_lines, parse_data_lines, read_export

MOCK_FILE_PATH = os.path.join(os.path.dirname(__file__), "test_nat_090.txt")

def test_extract_lines():
    lines = extract_lines(MOCK_FILE_PATH)
    assert len(lines) > 0, "Extracted lines should not be empty"
    assert isinstance(lines, list), "Extracted lines should be a list"
    assert all(isinstance(line, str) for line in lines), "All extracted lines should be strings"

def test_split_lines():
    lines = extract_lines(MOCK_FILE_PATH)
    meta, cols, data = split_lines(lines)

    assert isinstance(meta, list), "Metadata should be a list"
    assert isinstance(cols, str), "Header line should be a string"
    assert isinstance(data, list), "Data lines should be a list"
    
    assert len(meta) > 0, "Metadata lines should not be empty"
    assert len(cols) > 0, "Header line should not be empty"
    assert len(data) > 0, "Data lines should not be empty"


def test_parse_metadata_lines():
    lines = extract_lines(MOCK_FILE_PATH)
    meta, _, _ = split_lines(lines)
    metadata = parse_metadata_lines(meta)

    assert isinstance(metadata, dict), "Metadata should be a dictionary"
    assert len(metadata) > 0, "Metadata dictionary should not be empty"
    for key, value in metadata.items():
        assert isinstance(key, str), f"Metadata key '{key}' should be a string"
        assert isinstance(value, (str, int, float)), f"Metadata value for key '{key}' should be a string, int, or float"

def test_parse_data_lines():
    lines = extract_lines(MOCK_FILE_PATH)
    _, cols, data = split_lines(lines)
    df = parse_data_lines(data, header_line=cols)

    assert not df.empty, "DataFrame should not be empty"
    assert all(isinstance(col, str) for col in df.columns), "All DataFrame columns should be strings"
def test_read_export():
    raman = read_export(MOCK_FILE_PATH)

    assert "meta" in raman, "Output should contain 'meta' key"
    assert "data" in raman, "Output should contain 'data' key"
    metadata = raman["meta"]
    data_df = raman["data"]

    assert isinstance(metadata, dict), "Metadata should be a dictionary"
    assert len(metadata) > 0, "Metadata should not be empty"
    for key, value in metadata.items():
        assert isinstance(key, str), f"Metadata key '{key}' should be a string"
        assert isinstance(value, (str, int, float)), f"Metadata value for key '{key}' should be a string, int, or float"