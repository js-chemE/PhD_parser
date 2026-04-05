import os

from phd_parser.massspec.quadstar import extract_lines, split_lines, parse_metadata_lines, parse_data_lines, read_export

MOCK_FILE_PATH = os.path.join(os.path.dirname(__file__), "2026-03-26_sk1002_02-react.asc")

def test_extract_lines():
    lines = extract_lines(MOCK_FILE_PATH)
    assert len(lines) > 0, "Extracted lines should not be empty"
    assert isinstance(lines, list), "Extracted lines should be a list"
    assert all(isinstance(line, str) for line in lines), "All extracted lines should be strings"

def test_split_lines():
    lines = extract_lines(MOCK_FILE_PATH)
    meta_0, meta_1, meta_blocks, data = split_lines(lines)

    assert isinstance(meta_0, list), "Metadata block 0 should be a list"
    assert isinstance(meta_1, list), "Metadata block 1 should be a list"
    assert isinstance(meta_blocks, list), "Metadata blocks should be a list"
    assert isinstance(data, list), "Data lines should be a list"
    
    assert len(meta_0) > 0, "Metadata block 0 should not be empty"
    assert len(meta_1) > 0, "Metadata block 1 should not be empty"
    assert len(meta_blocks) > 0, "Metadata blocks list should not be empty"
    assert len(data) > 0, "Data lines should not be empty"

def test_parse_metadata_lines():
    lines = extract_lines(MOCK_FILE_PATH)
    meta_0, meta_1, meta_blocks, _ = split_lines(lines)
    metadata = parse_metadata_lines(meta_0, meta_1, meta_blocks)

    assert isinstance(metadata, dict), "Metadata should be a dictionary"
    assert len(metadata) > 0, "Metadata dictionary should not be empty"
    for key, value in metadata.items():
        assert isinstance(key, str), f"Metadata key '{key}' should be a string"
        assert isinstance(value, (str, int, float, dict)), f"Metadata value for key '{key}' should be a string, int, float or dict"

def test_parse_data_lines():
    lines = extract_lines(MOCK_FILE_PATH)
    _, _, _, data = split_lines(lines)
    df = parse_data_lines(data)

    assert not df.empty, "DataFrame should not be empty"
    assert all(isinstance(col, str) for col in df.columns), "All DataFrame columns should be strings"

def test_read_export():
    meta, df = read_export(MOCK_FILE_PATH)

    assert isinstance(meta, dict), "Metadata should be a dictionary"
    assert len(meta) > 0, "Metadata should not be empty"
    for key, value in meta.items():
        assert isinstance(key, str), f"Metadata key '{key}' should be a string"
        assert isinstance(value, (str, int, float, dict)), f"Metadata value for key '{key}' should be a string, int, float or dict"

    assert not df.empty, "DataFrame should not be empty"
    assert all(isinstance(col, str) for col in df.columns), "All DataFrame columns should be strings"
