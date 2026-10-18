"""Renishaw WiRe readers: the WDF binary and the plain-text export of the same measurement agree."""
import json
from pathlib import Path

import numpy as np
import pytest

from phd_parser.spectroscopy import RamanData, SpectroscopyData
from phd_parser.spectroscopy.raman import renishaw

FIXTURES = Path(__file__).parent / "renishaw"
WDF_532 = FIXTURES / "TiO2_Julius_532_1acc_01.wdf"
TXT_532 = FIXTURES / "TiO2_Julius_532_1acc_01.txt"
WDF_785 = FIXTURES / "TiO2_Julius_785_1acc_01.wdf"
TXT_785 = FIXTURES / "TiO2_Julius_785_1acc_01.txt"


def test_read_export_txt_keeps_every_row():
    raw = renishaw.read_export_txt(TXT_532)
    with open(TXT_532, encoding="utf-8") as handle:
        n_rows = sum(1 for line in handle if line.strip()) - 1  # minus the header
    assert len(raw["data"]) == n_rows
    assert list(raw["data"].columns) == ["wavenumber", "intensity"]
    assert raw["data"]["wavenumber"].dtype == float
    assert raw["meta"]["filename"] == TXT_532.name


def test_read_export_wdf_is_a_plain_result():
    wdf = renishaw.read_export_wdf(WDF_532)
    assert not isinstance(wdf, BaseException)
    assert wdf.x_unit == renishaw.UnitType.RamanShift
    assert wdf.data_unit == renishaw.UnitType.Counts
    assert wdf.wavenumber.ndim == 1 and wdf.data.shape[-1] == wdf.wavenumber.size
    assert 1e7 / wdf.laser_cm1 == pytest.approx(532.0, abs=1.0)
    assert wdf.acq_time is not None
    metadata = renishaw.wdf_metadata(wdf)
    json.dumps(metadata)  # JSON-safe, so it survives NetCDF
    assert metadata["scan_type"] == wdf.scan_type.name


@pytest.mark.parametrize("wdf_path, txt_path, excitation", [(WDF_532, TXT_532, 532.0), (WDF_785, TXT_785, 785.0)])
def test_wdf_and_txt_export_of_one_measurement_agree(wdf_path, txt_path, excitation):
    from_wdf = RamanData.from_renishaw_wdf(wdf_path)
    from_txt = RamanData.from_renishaw_txt(txt_path, excitation_wavelength_nm=excitation)
    assert type(from_wdf) is type(from_txt) is SpectroscopyData
    assert from_wdf.technique == from_txt.technique == "raman"
    assert from_wdf.excitation_wavelength_nm == pytest.approx(excitation, abs=1.0)
    assert from_wdf.n_spectral == from_txt.n_spectral
    np.testing.assert_allclose(np.sort(from_wdf.shift_per_cm), np.sort(from_txt.shift_per_cm), rtol=1e-5)
    np.testing.assert_allclose(from_wdf.sort().values, from_txt.sort().values, rtol=1e-5)


def test_from_renishaw_wdf_keeps_time_and_metadata(tmp_path):
    raman = RamanData.from_renishaw_wdf(WDF_532)
    assert raman.data_type == "counts"
    assert raman.tos_start is not None
    assert raman.metadata["instrument"] == "Renishaw"
    assert raman.metadata["acquisition_time"] == raman.tos_start.isoformat()
    assert raman.history[-1]["step"] == "from_renishaw_wdf"
    raman.to_netcdf(tmp_path / "raman.nc")
    back = SpectroscopyData.from_netcdf(tmp_path / "raman.nc")
    assert back.metadata == raman.metadata
    assert back.excitation_wavelength_nm == raman.excitation_wavelength_nm
