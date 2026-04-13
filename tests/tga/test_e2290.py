import os
import pandas as pd


MOCK_FILE_PATH = os.path.join(os.path.dirname(__file__), r"e2290/2025-08-29_JS0021_30-1000C_R5_100N2.txt")
MOCK_BLA_FILE_PATH = os.path.join(os.path.dirname(__file__), r"e2290/2025-09-03_blank_30-1000C_R5_100N2.txt")

def test_extract_lines():
    from phd_parser.tga.e2290 import extract_lines
    lines = extract_lines(MOCK_FILE_PATH)
    assert len(lines) > 0, "Extracted lines should not be empty"
    assert isinstance(lines, list), "Extracted lines should be a list"
    assert all(isinstance(line, str) for line in lines), "All extracted lines should be strings"

def test_split_sections():
    from phd_parser.tga.e2290 import extract_lines, split_sections
    lines = extract_lines(MOCK_FILE_PATH)
    header, data, results, sample, method = split_sections(lines)

    assert isinstance(header, list), "Header should be a list"
    assert isinstance(data, list), "Data should be a list"
    assert isinstance(results, list), "Results should be a list"
    assert isinstance(sample, list), "Sample should be a list"
    assert isinstance(method, list), "Method should be a list"

    assert len(header) > 0, "Header should not be empty"
    assert len(data) > 0, "Data should not be empty"
    assert len(results) > 0, "Results should not be empty"
    assert len(sample) > 0, "Sample should not be empty"
    assert len(method) > 0, "Method should not be empty"

def test_parse_data():
    from phd_parser.tga.e2290 import extract_lines, split_sections, parse_data
    lines = extract_lines(MOCK_FILE_PATH)
    _, data_lines, _, _, _ = split_sections(lines)
    df, units = parse_data(data_lines)

    assert not df.empty, "DataFrame should not be empty"
    assert all(isinstance(col, str) for col in df.columns), "All DataFrame columns should be strings"
    assert len(units) == len(df.columns), "Units length should match number of DataFrame columns"

def test_read_tga_e2290():
    from phd_parser.tga.e2290 import read_tga_e2290
    tga_data = read_tga_e2290(MOCK_FILE_PATH)

    assert isinstance(tga_data, dict), "Output should be a dictionary"
    assert "curve_name" in tga_data, "Output should contain 'curve_name' key"
    assert "saved_at" in tga_data, "Output should contain 'saved_at' key"
    assert "performed_at" in tga_data, "Output should contain 'performed_at' key"
    assert "data" in tga_data, "Output should contain 'data' key"
    assert "units" in tga_data, "Output should contain 'units' key"
    assert "results" in tga_data, "Output should contain 'results' key"
    assert "sample_name" in tga_data, "Output should contain 'sample_name' key"
    assert "weight" in tga_data, "Output should contain 'weight' key"
    assert "method" in tga_data, "Output should contain 'method' key"

    assert isinstance(tga_data["data"], pd.DataFrame), "'data' should be a DataFrame"