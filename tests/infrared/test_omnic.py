import os
from pathlib import Path

MOCK_DIR_PATH = Path(os.path.join(os.path.dirname(__file__))) / "omnic-test-data"
MOCK_SINGLE_FILE_PATH = MOCK_DIR_PATH / "Spectrum Index 2718 at 2,46 Hours.spa"

def test_read_spa_single():
    from phd_parser.infrared.omnic import read_spa
    path = MOCK_SINGLE_FILE_PATH
    raw = read_spa(path)
    x = raw["data"]["x"]
    v = raw["data"]["v"]
    meta = raw["meta"]
    assert x is not None, "X data should not be None"
    assert v is not None, "V (intensity) data should not be None"
    assert x.ndim == 1, f"X data should be 1D, but got shape {x.shape}"
    assert v.ndim == 1, f"V data should be 1D for a single spectrum, but got shape {v.shape}"
    assert v.size == x.size, f"X and V data should have the same number of points, but got X size {x.size} and V size {v.size}"
    assert meta is not None, "Metadata should not be None"

def test_read_spa_dir():
    from phd_parser.infrared.omnic import read_spa
    raw = read_spa(MOCK_DIR_PATH)
    x = raw["data"]["x"]
    v = raw["data"]["v"]
    meta = raw["meta"]
    assert x is not None, "X data should not be None"
    assert v is not None, "V (intensity) data should not be None"
    assert v.ndim == 2, f"V data should be 2D (multiple spectra), but got shape {v.shape}"
    assert v.shape[1] == x.shape[0], f"X data length should match the number of points per spectrum in V data, but got X length {x.shape[0]} and V shape {v.shape}"
    assert meta is not None, "Metadata should not be None"

def test_read_spa_single_irdata():
    from phd_parser.infrared import IRData
    path = MOCK_SINGLE_FILE_PATH
    ir_data = IRData.from_omnic_spa(path)

    assert ir_data.wavenumber is not None, "Wavenumber data should not be None"
    assert ir_data.values is not None, "Intensity data should not be None"
    assert ir_data.ndim == 1, f"Data should be 1D (single spectrum), but got shape {ir_data.shape}"
    assert ir_data.values.size == ir_data.wavenumber.size, (
        f"Wavenumber and intensity data should have the same number of points, "
        f"but got wavenumber size {ir_data.wavenumber.size} and values size {ir_data.values.size}"
    )
    assert ir_data.ds.attrs.get("data_type") is not None, "data_type attribute should not be None"

def test_read_spa_dir_irdata():
    from phd_parser.infrared import IRData
    ir_data = IRData.from_omnic_spa(MOCK_DIR_PATH)

    assert ir_data.wavenumber is not None, "Wavenumber data should not be None"
    assert ir_data.values is not None, "Intensity data should not be None"
    assert ir_data.ndim == 2, f"Data should be 2D (multiple spectra), but got shape {ir_data.shape}"
    assert ir_data.shape[1] == ir_data.wavenumber.size, (
        f"Wavenumber length should match the number of points per spectrum, "
        f"but got wavenumber length {ir_data.wavenumber.size} and shape {ir_data.shape}"
    )
    assert ir_data.ds.attrs.get("data_type") is not None, "data_type attribute should not be None"
