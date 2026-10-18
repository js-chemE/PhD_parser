"""XPSData: the raw VAMAS and Avantage files of the Thermo instrument, the CasaXPS export, and processing."""
from pathlib import Path

import numpy as np
import pytest

from phd_parser.spectroscopy.xps import XPSData, avantage, vamas

FIXTURES = Path(__file__).parent
VMS_MULTI = FIXTURES / "vamas" / "SK-0001_3.vms"
VMS_SINGLE_REGION_FILE = FIXTURES / "vamas" / "C1s Scan.vms"
AVG_C1S = FIXTURES / "avantage" / "C1s Scan.avg"
AVG_SURVEY = FIXTURES / "avantage" / "XPS Survey.avg"
CASA_SINGLE = FIXTURES / "casaxps_export_single.txt"
CASA_DOUBLET = FIXTURES / "casaxps_export_doublet.txt"


def test_read_vamas_follows_the_standard():
    raw = vamas.read_vamas(VMS_MULTI)
    header = raw["meta"]["header"]
    assert header["experiment_mode"] == "NORM" and header["scan_mode"] == "REGULAR"
    assert header["n_blocks"] == len(raw["data"]) == 6
    assert "C1s Scan" in raw["data"] and "XPS Survey" in raw["data"]
    block = raw["data"]["C1s Scan"]
    meta = raw["meta"]["blocks"]["C1s Scan"]
    assert meta["technique"] == "XPS" and meta["species_label"] == "C 1s"
    assert meta["source_energy_eV"] == pytest.approx(1486.68)
    assert meta["pass_energy_eV"] == 50 and meta["n_scans"] == 10 and meta["dwell_time_s"] == pytest.approx(0.05)
    assert block["variables"] == ["Counts", "Transmission"]
    assert block["kinetic_energy"][0] == pytest.approx(1188.6) and block["counts"].size == 191
    assert meta["datetime"].year == 2026 and meta["sample"] == "SK_0001_3"


def test_from_vamas_gives_one_region_per_block():
    regions = XPSData.from_vamas(VMS_MULTI)
    assert set(regions) == {"C1s Scan", "Mo3d Scan", "N1s Scan", "O1s Scan", "Ti2p Scan", "XPS Survey"}
    c1s = regions["C1s Scan"]
    assert c1s.region == "C1s Scan" and c1s.sample == "SK_0001_3" and c1s.species_label == "C 1s"
    assert c1s.data_type == "counts" and c1s.n_points == 191
    # binding energy = source energy - kinetic energy, stored ascending
    np.testing.assert_allclose(c1s.binding_energy[0], 1486.68 - 1207.6, atol=1e-3)
    np.testing.assert_allclose(c1s.kinetic_energy, 1486.68 - c1s.binding_energy)
    assert c1s.transmission is not None and c1s.transmission.shape == c1s.values.shape
    assert c1s.acquisition_time is not None
    assert c1s.history[-1]["step"] == "_from_vamas_block"
    assert XPSData.from_vamas(VMS_MULTI, region="O1s Scan").region == "O1s Scan"
    with pytest.raises(KeyError):
        XPSData.from_vamas(VMS_MULTI, region="Pt4f Scan")


def test_from_avantage_avg_agrees_with_the_vamas_axis():
    avg = XPSData.from_avantage_avg(AVG_C1S)
    vms = XPSData.from_vamas(VMS_MULTI, region="C1s Scan")
    assert avg.region == "C1s Scan" and avg.n_points == vms.n_points
    np.testing.assert_allclose(avg.binding_energy, vms.binding_energy, atol=1e-3)  # 1486.680054 vs 1486.68
    assert avg.source_energy_eV == pytest.approx(1486.68, abs=1e-3)
    assert avg.dwell_time_s == pytest.approx(0.05) and avg.n_scans == 10
    # Avantage had the transmission function removed: not raw counts any more
    assert avg.ds.attrs["transmission_removed"] is True and avg.data_type == "arbitrary"
    assert avg.acquisition_time is not None
    survey = XPSData.from_avantage_avg(AVG_SURVEY)
    assert survey.region == "XPS Survey" and survey.binding_energy.max() > 1000


def test_read_avg_parses_properties_and_axis():
    raw = avantage.read_avg(AVG_C1S)
    assert raw["meta"]["axis"]["unit"] == "eV" and raw["meta"]["axis"]["n_points"] == 191
    assert raw["meta"]["DS_ANPROPID_LENS_MODE_NAME"] == "Standard"
    assert raw["data"]["values"].size == 191


def test_from_casaxps_export_keeps_the_fit():
    single = XPSData.from_casaxps_export(CASA_SINGLE)
    assert single.region == "C1s Scan" and single.sample == "SK_0001_1"
    assert single.data_type == "counts" and single.source_energy_eV == pytest.approx(1486.68)
    assert single.has_fit and single.has_background and single.envelope is not None
    assert list(single.peaks["name"]) == ["C 1s"]
    assert single.peaks["position_eV"].iloc[0] == pytest.approx(284.81)
    assert single.ds.attrs["analysis_source"] == "CasaXPS"
    doublet = XPSData.from_casaxps_export(CASA_DOUBLET)
    assert list(doublet.peaks["name"]) == ["Mo 3d5/2", "Mo 3d3/2"]
    assert doublet.ds["peaks"].shape == (2, doublet.n_points)
    # counts / acquisition time is what CasaXPS calls CPS
    cps = doublet.to_cps() if doublet.dwell_time_s else None
    assert cps is None  # the export has no dwell time and scan count, only their product
    assert doublet.ds.attrs["acquisition_time_per_point_s"] == pytest.approx(0.5)


def test_processing_is_immutable_and_recorded():
    region = XPSData.from_vamas(VMS_MULTI, region="C1s Scan")
    cps = region.to_cps()
    np.testing.assert_allclose(cps.values, region.values / (0.05 * 10))
    np.testing.assert_allclose(cps.to_counts().values, region.values)
    shifted = region.calibrate_binding_energy(284.8, 285.3)
    np.testing.assert_allclose(shifted.binding_energy, region.binding_energy - 0.5)
    assert shifted.ds.attrs["binding_energy_shift_eV"] == pytest.approx(-0.5)
    assert shifted.shift_binding_energy(0.2).ds.attrs["binding_energy_shift_eV"] == pytest.approx(-0.3)
    window = region.select_binding_energy_range(280, 292)
    assert window.n_points < region.n_points and window.binding_energy.min() >= 280
    assert [entry["step"] for entry in window.history] == ["_from_vamas_block", "select_binding_energy_range"]
    assert region.n_points == 191  # untouched
    smoothed = region.smooth_savgol(7, 2)
    assert smoothed.values.shape == region.values.shape
    assert region.normalise_max().values.max() == pytest.approx(1.0)


def test_shirley_background_is_recovered_from_a_peak_that_sits_on_one():
    energy = np.linspace(280.0, 295.0, 151)
    peak = 1000.0 * np.exp(-((energy - 285.0) / 0.8) ** 2)
    # a Shirley-shaped step: proportional to the integrated peak towards high binding energy
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (peak[1:] + peak[:-1]) * np.diff(energy))])
    step = 200.0 + 100.0 * cumulative / cumulative[-1]
    region = XPSData.from_arrays(energy, peak + step, source_energy_eV=1486.68, region="test")
    corrected = region.subtract_background("shirley")
    assert corrected.has_background and corrected.ds.attrs["background_method"] == "shirley"
    np.testing.assert_allclose(corrected.background, step, atol=0.5)
    np.testing.assert_allclose(corrected.values, peak, atol=0.5)
    np.testing.assert_allclose(corrected.values + corrected.background, region.values)
    assert corrected.integrate(282, 288) == pytest.approx(np.trapezoid(peak, energy), rel=0.01)
    linear = region.subtract_background("linear", min_eV=282, max_eV=288)
    assert linear.background[0] == 0.0  # outside the window nothing is subtracted
    assert linear.background[np.searchsorted(energy, 282.0)] == pytest.approx(region.values[np.searchsorted(energy, 282.0)])


def test_netcdf_round_trip_keeps_everything(tmp_path):
    region = XPSData.from_casaxps_export(CASA_DOUBLET).shift_binding_energy(0.3)
    region.to_netcdf(tmp_path / "mo3d.nc")
    back = XPSData.from_netcdf(tmp_path / "mo3d.nc")
    np.testing.assert_allclose(back.values, region.values)
    np.testing.assert_allclose(back.binding_energy, region.binding_energy)
    assert back.has_fit and list(back.peaks["name"]) == ["Mo 3d5/2", "Mo 3d3/2"]
    assert back.metadata == region.metadata
    assert [entry["step"] for entry in back.history][-2:] == ["shift_binding_energy", "from_netcdf"]
    assert back.ds.attrs["binding_energy_shift_eV"] == pytest.approx(0.3)
    frame = back.to_dataframe()
    assert "kinetic_energy_eV" in frame.columns and frame.index.name == "binding_energy_eV"
