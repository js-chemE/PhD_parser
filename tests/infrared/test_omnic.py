import os
from pathlib import Path

MOCK_DIR_PATH = Path(os.path.join(os.path.dirname(__file__))) / "omnic-test-data"
MOCK_SINGLE_FILE_PATH = MOCK_DIR_PATH / "Spectrum Index 2718 at 2,46 Hours.spa"

def test_read_spa_single():
    from phd_parser.infrared.omnic import read_spa
    path = MOCK_SINGLE_FILE_PATH
    raw = read_spa(path)
    x = raw["data"]["x"]
    y = raw["data"]["y"]
    meta = raw["meta"]
    assert x is not None, "X data should not be None"
    assert y is not None, "Y data should not be None"
    assert x.ndim == 1, f"X data should be 1D, but got shape {x.shape}"
    assert y.ndim == 1, f"Y data should be 2D (even for single spectrum), but got shape {y.shape}"
    assert y.size == x.size, f"X and Y data should have the same number of points, but got X size {x.size} and Y size {y.size}"
    assert meta is not None, "Metadata should not be None"

def test_read_spa_dir():
    from phd_parser.infrared.omnic import read_spa
    raw = read_spa(MOCK_DIR_PATH)
    x = raw["data"]["x"]
    y = raw["data"]["y"]
    meta = raw["meta"]
    assert x is not None, "X data should not be None"
    assert y is not None, "Y data should not be None"
    assert y.ndim == 2, f"Y data should be 2D (multiple spectra), but got shape {y.shape}"
    assert y.shape[1] == x.shape[0], f"X data length should match the number of spectra in Y data, but got X length {x.shape[0]} and Y shape {y.shape}"
    assert meta is not None, "Metadata should not be None"

def test_read_spa_single_irdata():
    from phd_parser.infrared import IRData
    path = MOCK_SINGLE_FILE_PATH
    ir_data = IRData.from_omnic_spa(path)

    assert ir_data.x is not None, "X data should not be None"
    assert ir_data.y is not None, "Y data should not be None"
    assert ir_data.y.ndim == 1, f"Y data should be 1D (even for single spectrum), but got shape {ir_data.y.shape}"
    assert ir_data.y.size == ir_data.x.size, f"X and Y data should have the same number of points, but got X size {ir_data.x.size} and Y size {ir_data.y.size}"
    assert ir_data.raw_meta is not None, "Raw metadata should not be None"

def test_read_spa_dir_irdata():
    from phd_parser.infrared import IRData
    ir_data = IRData.from_omnic_spa(MOCK_DIR_PATH)

    assert ir_data.x is not None, "X data should not be None"
    assert ir_data.y is not None, "Y data should not be None"
    assert ir_data.y.ndim == 2, f"Y data should be 2D (multiple spectra), but got shape {ir_data.y.shape}"
    assert ir_data.y.shape[1] == ir_data.x.shape[0], f"X data length should match the number of spectra in Y data, but got X length {ir_data.x.shape[0]} and Y shape {ir_data.y.shape}"
    assert ir_data.raw_meta is not None, "Raw metadata should not be None"