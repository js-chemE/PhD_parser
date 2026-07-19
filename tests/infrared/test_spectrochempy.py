import os
from pathlib import Path

import numpy as np
import pytest

MOCK_DIR_PATH = Path(os.path.dirname(__file__)) / "omnic-test-data"
MOCK_SINGLE_FILE_PATH = MOCK_DIR_PATH / "Spectrum Index 2718 at 2,46 Hours.spa"

pytest.importorskip("spectrochempy", reason="spectrochempy not installed")


# ----------------------------------------------------------------
# spectrochempy — raw loading
# ----------------------------------------------------------------

def test_read_omnic_single_file():
    import spectrochempy as scp

    nd = scp.read_omnic(MOCK_SINGLE_FILE_PATH)
    assert nd.data is not None
    # spectrochempy always returns 2-D (y: n_files, x: n_wavenumber)
    assert nd.data.ndim == 2
    assert nd.data.shape[0] == 1
    assert nd.x is not None
    assert nd.x.data.size == nd.data.shape[1]


def test_read_omnic_directory():
    import spectrochempy as scp

    files = sorted(MOCK_DIR_PATH.glob("*.spa"))
    nd = scp.read_omnic(*files)
    assert nd.data.ndim == 2
    assert nd.data.shape[0] == len(files)
    assert nd.data.shape[1] == nd.x.data.size


def test_read_omnic_single_wavenumber_axis_units():
    import spectrochempy as scp

    nd = scp.read_omnic(MOCK_SINGLE_FILE_PATH)
    assert "cm" in str(nd.x.units)


def test_read_omnic_data_type_attribute():
    import spectrochempy as scp

    nd = scp.read_omnic(MOCK_SINGLE_FILE_PATH)
    # Title holds the spectral quantity (e.g. "Absorbance")
    assert nd.title is not None
    assert len(nd.title) > 0


# ----------------------------------------------------------------
# Cross-validation against IRData.from_omnic_spa
# ----------------------------------------------------------------

def test_wavenumber_axis_agrees_with_phd_parser():
    import spectrochempy as scp

    from phd_parser.infrared import IRData

    nd = scp.read_omnic(MOCK_SINGLE_FILE_PATH)
    ir = IRData.from_omnic_spa(MOCK_SINGLE_FILE_PATH)

    # spectrochempy: wavenumber in cm⁻¹; phd_parser: m⁻¹ (divide by 100 to get cm⁻¹)
    scp_wn = np.sort(nd.x.data.squeeze())
    phd_wn = np.sort(ir.wavenumber_per_cm)

    assert scp_wn.size == phd_wn.size
    np.testing.assert_allclose(scp_wn, phd_wn, rtol=1e-4)


def test_spectral_values_agree_with_phd_parser():
    import spectrochempy as scp

    from phd_parser.infrared import IRData

    nd = scp.read_omnic(MOCK_SINGLE_FILE_PATH)
    ir = IRData.from_omnic_spa(MOCK_SINGLE_FILE_PATH)

    # Sort both by ascending wavenumber before comparing
    scp_wn_order = np.argsort(nd.x.data.squeeze())
    phd_wn_order = np.argsort(ir.wavenumber_per_cm)

    scp_values = nd.data.squeeze()[scp_wn_order]
    phd_values = ir.values[phd_wn_order]

    np.testing.assert_allclose(scp_values, phd_values, rtol=1e-4)


def test_scan_count_agrees_with_phd_parser():
    import spectrochempy as scp

    from phd_parser.infrared import IRData

    files = sorted(MOCK_DIR_PATH.glob("*.spa"))
    nd = scp.read_omnic(*files)
    ir = IRData.from_omnic_spa(MOCK_DIR_PATH)

    assert nd.data.shape[0] == ir.shape[0]
    assert nd.data.shape[1] == ir.shape[1]


# ----------------------------------------------------------------
# IRData.from_omnic_spa — backend parameter
# ----------------------------------------------------------------

def test_from_omnic_spa_backend_spectrochempy_single():
    from phd_parser.infrared import IRData

    ir = IRData.from_omnic_spa(MOCK_SINGLE_FILE_PATH, backend="spectrochempy")
    assert ir.ndim == 1
    assert ir.data_type is not None
    assert ir.wavenumber.size > 0


def test_from_omnic_spa_backend_spectrochempy_directory():
    from phd_parser.infrared import IRData

    ir = IRData.from_omnic_spa(MOCK_DIR_PATH, backend="spectrochempy")
    assert ir.ndim == 2
    assert ir.shape[0] == len(list(MOCK_DIR_PATH.glob("*.spa")))


def test_from_omnic_spa_backend_omnic_single():
    from phd_parser.infrared import IRData

    ir = IRData.from_omnic_spa(MOCK_SINGLE_FILE_PATH, backend="omnic")
    assert ir.ndim == 1
    assert ir.data_type is not None


def test_from_omnic_spa_backends_produce_same_result():
    from phd_parser.infrared import IRData

    ir_scp = IRData.from_omnic_spa(MOCK_SINGLE_FILE_PATH, backend="spectrochempy")
    ir_omnic = IRData.from_omnic_spa(MOCK_SINGLE_FILE_PATH, backend="omnic")

    np.testing.assert_allclose(
        np.sort(ir_scp.wavenumber_per_cm), np.sort(ir_omnic.wavenumber_per_cm), rtol=1e-4
    )
    sort_scp = np.argsort(ir_scp.wavenumber_per_cm)
    sort_omnic = np.argsort(ir_omnic.wavenumber_per_cm)
    np.testing.assert_allclose(
        ir_scp.values[sort_scp], ir_omnic.values[sort_omnic], rtol=1e-4
    )
    assert ir_scp.data_type == ir_omnic.data_type


def test_from_omnic_spa_auto_uses_spectrochempy_when_available():
    from phd_parser.infrared import IRData

    ir_auto = IRData.from_omnic_spa(MOCK_SINGLE_FILE_PATH, backend="auto")
    ir_scp = IRData.from_omnic_spa(MOCK_SINGLE_FILE_PATH, backend="spectrochempy")

    np.testing.assert_allclose(
        np.sort(ir_auto.wavenumber_per_cm), np.sort(ir_scp.wavenumber_per_cm), rtol=1e-4
    )


# ----------------------------------------------------------------
# IRData.from_scp
# ----------------------------------------------------------------

def test_from_scp_single_spectrum():
    import spectrochempy as scp
    from phd_parser.infrared import IRData

    nd = scp.read_omnic(MOCK_SINGLE_FILE_PATH)
    ir = IRData.from_scp(nd)

    assert ir.ndim == 1
    assert ir.data_type is not None
    assert ir.wavenumber.size > 0
    assert ir.values.size == ir.wavenumber.size


def test_from_scp_multi_scan():
    import spectrochempy as scp
    from phd_parser.infrared import IRData

    files = sorted(MOCK_DIR_PATH.glob("*.spa"))
    nd = scp.read_omnic(*files)
    ir = IRData.from_scp(nd)

    assert ir.ndim == 2
    assert ir.shape[0] == len(files)
    assert ir.shape[1] == ir.wavenumber.size


def test_from_scp_wavenumber_in_si_units():
    import spectrochempy as scp
    from phd_parser.infrared import IRData

    nd = scp.read_omnic(MOCK_SINGLE_FILE_PATH)
    ir = IRData.from_scp(nd)

    # wavenumber stored in m⁻¹; typical mid-IR range ~40 000–400 000 m⁻¹
    assert ir.wavenumber.min() > 1e4
    assert ir.wavenumber.max() < 1e6


def test_from_scp_matches_from_omnic_spa():
    import spectrochempy as scp
    from phd_parser.infrared import IRData

    nd = scp.read_omnic(MOCK_SINGLE_FILE_PATH)
    ir_scp = IRData.from_scp(nd)
    ir_omnic = IRData.from_omnic_spa(MOCK_SINGLE_FILE_PATH, backend="omnic")

    sort_scp = np.argsort(ir_scp.wavenumber_per_cm)
    sort_omnic = np.argsort(ir_omnic.wavenumber_per_cm)
    np.testing.assert_allclose(
        ir_scp.wavenumber_per_cm[sort_scp], ir_omnic.wavenumber_per_cm[sort_omnic], rtol=1e-4
    )
    np.testing.assert_allclose(
        ir_scp.values[sort_scp], ir_omnic.values[sort_omnic], rtol=1e-4
    )
    assert ir_scp.data_type == ir_omnic.data_type


def test_from_scp_delta_time_seconds():
    import spectrochempy as scp
    from phd_parser.infrared import IRData

    files = sorted(MOCK_DIR_PATH.glob("*.spa"))
    nd = scp.read_omnic(*files)
    ir = IRData.from_scp(nd, delta_time_seconds=30.0)

    assert ir.tos is not None
    assert ir.tos[0] == pytest.approx(0.0)
    np.testing.assert_allclose(np.diff(ir.tos), 30.0)
