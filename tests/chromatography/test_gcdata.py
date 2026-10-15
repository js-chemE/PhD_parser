"""GCData: the vendor-independent chromatography container.

Built from the one EZChrom injection in ``agilent/ezchrom-test-data`` and from
synthetic records for the axis cases real data rarely shows.
"""
from pathlib import Path

import numpy as np
import pytest

from phd_parser.chromatography import GCData, ezchrom, group_injections

RESULT_SET = Path(__file__).parent / "agilent" / "ezchrom-test-data" / "2025-05-15 16-10-24 (GMT +02-00).rslt"


@pytest.fixture(scope="module")
def gc():
    return GCData.from_ezchrom_rslt(RESULT_SET)


def _record(time, interval=0.05, points=100, detectors=("fid",), offset=0.0):
    axis = offset + np.arange(points) * interval
    return dict(
        injection_time=time,
        detectors={name: dict(retention_time=axis, values=np.arange(points, dtype=float), unit="pA",
                              vendor_name=name, scale_factor=None) for name in detectors},
        peaks=[dict(detector=detectors[0], species="CO", retention_time=1.0, area=2.0, height=1.0)],
        species=[dict(detector=detectors[0], species="CO", amount=0.5, amount_unit="%", area=2.0, height=1.0)],
    )


def test_ezchrom_injection_becomes_one_injection(gc):
    assert gc.n_injections == 1
    assert gc.detectors == ["fid", "tcd"]
    assert gc.get_unit("fid") == "pA"
    assert gc.analysis_source.startswith("OpenLab CDS EZChrom")
    assert gc.ds.attrs["tos_start"] == "2025-05-15T14:10:51+00:00"


def test_chromatogram_is_the_parsed_signal(gc):
    parsed = ezchrom.read_dat(next(RESULT_SET.glob("*.dat")))["data"]
    one = gc.get_chromatogram("tcd", injection=0)
    assert one.ndim == 1
    np.testing.assert_array_equal(one.values, parsed["Back Signal"]["signal"])
    np.testing.assert_allclose(gc.get_retention_time("tcd"), parsed["Back Signal"]["time"], rtol=0, atol=1e-9)


def test_chromatogram_slices_like_numpy(gc):
    every = gc.get_chromatogram("fid")
    assert every.dims == ("injection", "retention_time_fid")
    window = every[0, 100:200]
    assert window.shape == (100,)
    assert window["retention_time_fid"].values[0] == pytest.approx(5.0)


def test_analysis_in_si_and_detector_units(gc):
    peaks = gc.get_peaks()
    methanol = peaks[peaks["species"] == "Methanol"].iloc[0]
    assert methanol["detector"] == "fid"
    assert methanol["retention_time"] == pytest.approx(6.75 * 60)
    assert methanol["area"] == pytest.approx(2012 / 7680, rel=1e-6)
    amounts = gc.get_amounts("tcd")
    assert amounts.loc[0, "CO2"] == pytest.approx(24.403, abs=1e-3)
    # CH4 is quantified on both detectors; the not-found one is placed from the method's peak table
    on = {detector for detector, species in gc.get_amounts().columns[1:] if species == "CH4"}
    assert on == {"fid", "tcd"}
    with pytest.raises(KeyError):
        gc.get_evolution("CH4")


def test_netcdf_round_trip_is_lossless_and_packed(gc, tmp_path):
    path = tmp_path / "gc.nc"
    gc.to_netcdf(path)
    back = GCData.from_netcdf(path)
    for name in gc.detectors:
        np.testing.assert_array_equal(back.ds[name].values, gc.ds[name].values)
        assert back.ds[name].attrs["scale_factor"] == pytest.approx(gc.ds[name].attrs["scale_factor"])
    assert back.get_peaks().equals(gc.get_peaks())
    assert isinstance(back.ds.attrs["calibration"], dict)
    assert back.tos_start == gc.tos_start


def test_different_run_lengths_are_padded():
    records = [_record("2025-01-01 10:00", points=100), _record("2025-01-01 10:20", points=80)]
    gc = GCData.from_injections(records)
    chromatograms = gc.get_chromatogram("fid").values
    assert chromatograms.shape == (2, 100)
    assert np.isnan(chromatograms[1, 80:]).all() and not np.isnan(chromatograms[1, :80]).any()
    assert gc.tos.tolist() == [0.0, 1200.0]


def test_detectors_may_have_different_axes():
    record = _record("2025-01-01 10:00", detectors=("fid",))
    record["detectors"]["tcd"] = dict(retention_time=np.arange(50) * 0.1, values=np.zeros(50), unit="uV")
    gc = GCData.from_injections([record])
    assert gc.get_retention_time("fid").size == 100 and gc.get_retention_time("tcd").size == 50


def test_changed_sampling_interval_is_grouped_not_resampled():
    records = [_record("2025-01-01 10:00"), _record("2025-01-01 10:20", interval=0.1), _record("2025-01-01 10:40")]
    assert group_injections(records) == [[0, 2], [1]]
    with pytest.raises(ValueError, match="group_injections"):
        GCData.from_injections(records)


def test_selection_keeps_the_matching_peaks():
    gc = GCData.from_injections([_record("2025-01-01 10:00"), _record("2025-01-01 10:20"), _record("2025-01-01 10:40")])
    second = gc.select_by_idx(1)
    assert second.n_injections == 1 and second.get_peaks()["injection"].tolist() == [1]
    assert gc.select_tos_range(1000, None).n_injections == 2
    assert gc.select_timestamp_range("2025-01-01 10:30").n_injections == 1


def test_time_origin():
    gc = GCData.from_injections([_record("2025-01-01 10:00"), _record("2025-01-01 10:20")])
    moved = gc.move_tos_start_by(600)
    assert moved.tos.tolist() == [-600.0, 600.0]
    assert (moved.timestamps == gc.timestamps).all()
    assert gc.set_tos_start("2025-01-01 09:00").tos.tolist() == [0.0, 1200.0]


def test_merge_appends_and_rebases():
    first = GCData.from_injections([_record("2025-01-01 10:00")])
    second = GCData.from_injections([_record("2025-01-01 11:00", points=90)])
    merged = first.merge(second)
    assert merged.n_injections == 2
    assert merged.tos.tolist() == [0.0, 3600.0]
    assert merged.get_peaks()["injection"].tolist() == [0, 1]
    assert merged.get_amounts("fid")["CO"].tolist() == [0.5, 0.5]


def test_from_arrays():
    gc = GCData.from_arrays(np.arange(10) * 0.5, np.ones(10), detector="tcd", unit="uV")
    assert gc.n_injections == 1 and not gc.has_analysis
    assert gc.get_chromatogram("tcd", injection=0).values.sum() == 10


def test_rename_species_to_the_standard(gc):
    renamed = gc.rename_species()
    assert "MeOH" in set(renamed.get_peaks()["species"]) and "Methanol" not in set(renamed.get_peaks()["species"])
    assert renamed.get_evolution("MeOH").values[0] == gc.get_evolution("Methanol").values[0]
    assert renamed.ds.attrs["species_convention"] == "phd_standard"
    assert "Methanol" in set(gc.get_peaks()["species"])  # the original is untouched


def test_rename_species_refuses_to_merge_two_species():
    record = _record("2025-01-01 10:00")
    record["species"].append(dict(detector="fid", species="carbon monoxide", amount=0.1, area=1.0, height=1.0))
    with pytest.raises(ValueError, match="merges"):
        GCData.from_injections([record]).rename_species()
