"""Agilent OpenLab EZChrom reader, checked against the instrument's own exports.

The test data is one injection of the 2025-05-15 Box 2 OP003 reduction
sequence: the ``.dat`` and ``.acam_`` from its ``.rslt`` folder, the sequence
and instrument containers, the acquisition method ACAML, and the AIA netCDF
export OpenLab wrote for the same injection.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import netcdf_file

from phd_parser.chromatography.agilent import ezchrom

DATA = Path(__file__).parent / "ezchrom-test-data"
RESULT_SET = DATA / "2025-05-15 16-10-24 (GMT +02-00).rslt"
DAT = RESULT_SET / "2025-05-15 16-10-34 (GMT +02-00)-Rep1.dat"
ACAM = RESULT_SET / "2025-05-15 16-10-24 (GMT +02-00).rst.0.0001.acam_"
NETCDF = {
    "Front Signal": DATA / "2025-05-15 16-10-34 (GMT +02-00)-Rep1dat-Front Signal.cdf",
    "Back Signal": DATA / "2025-05-15 16-10-34 (GMT +02-00)-Rep1dat-Back Signal.cdf",
}


@pytest.fixture(scope="module")
def dat():
    return ezchrom.read_dat(DAT)


@pytest.mark.parametrize("signal", ["Front Signal", "Back Signal"])
def test_trace_matches_netcdf_export(dat, signal):
    with netcdf_file(NETCDF[signal], "r", mmap=False) as handle:
        exported = handle.variables["ordinate_values"].data.astype(float)
    trace = dat["data"][signal]["signal"]
    # the export drops the last sample (an unacquired placeholder) and is float32
    assert len(trace) == len(exported) + 1
    np.testing.assert_allclose(trace[:-1], exported, rtol=0, atol=1e-6 * np.abs(exported).max())


def test_trace_is_raw_counts_on_a_regular_time_axis(dat):
    for trace in dat["data"].values():
        assert np.all(trace["counts"] == np.round(trace["counts"]))
        assert trace["scale_factor"] == pytest.approx(1 / 7680)
        assert trace["sampling_interval"] == 0.05
        assert trace["time"][0] == 0
        assert trace["time"][-1] == pytest.approx(17.2 * 60)
        assert trace["counts"][-1] == 0
    assert dat["data"]["Front Signal"]["unit"] == "pA"
    assert dat["data"]["Back Signal"]["unit"] == "25 µV"


def test_metadata(dat):
    meta = dat["meta"]
    assert meta["header"]["instrument_name"] == "pilot_plant_GC"
    assert meta["header"]["method_path"].endswith("JS_2025-05-12_CH.met")
    assert meta["summary"]["created"].startswith("2025-05-15T14:10:34")
    xml = meta["instrument_method_xml"]["GC7890_00"]
    assert xml.lstrip().startswith("<?xml") and "GCContextualMethod" in xml
    assert meta["audit_trail"][0]["description"] == "New data file created."
    assert "Run Time                                     17.2 min" in meta["instrument_setup"]["InstrumentSetupGeneral"]


def test_integrator_peak_is_reproduced_from_the_raw_counts(dat):
    peaks = ezchrom.read_acaml(ACAM)["data"]["injections"][0]["peaks"]
    methanol = next(peak for peak in peaks if peak["species"] == "Methanol")
    trace = dat["data"][methanol["signal"]]
    inside = (trace["time"] >= methanol["BeginTime"] * 60 - 1e-9) & (trace["time"] <= methanol["EndTime"] * 60 + 1e-9)
    above_baseline = trace["counts"][inside] - methanol["LevelStart"]
    assert above_baseline.max() == methanol["Height"]
    assert trace["time"][inside][np.argmax(above_baseline)] / 60 == pytest.approx(methanol["RetentionTime"])
    assert above_baseline.sum() * trace["sampling_interval"] == pytest.approx(methanol["Area"], rel=1e-3)


def test_acaml_results():
    parsed = ezchrom.read_acaml(ACAM)
    injection = parsed["data"]["injections"][0]
    assert injection["dat_file"] == DAT.name
    amounts = {record["species"]: record["Amount"] for record in injection["species"] if record["peak_id"]}
    assert amounts["CO2"] == pytest.approx(24.403, abs=1e-3)
    assert amounts["H2"] == pytest.approx(68.102, abs=1e-3)
    assert parsed["meta"]["instrument"]["module_SerialNo"] == "CN17043020"
    curve = next(iter(parsed["data"]["calibration_curves"].values()))
    assert curve["levels"][0]["history"]


def test_read_result_set():
    result_set = ezchrom.read_result_set(RESULT_SET)
    (injection,) = result_set["data"]["injections"]
    assert set(injection["data"]) == {"Front Signal", "Back Signal"}
    assert injection["peaks"] and injection["species"] and injection["calibration_curves"]
    assert "instrument_method_xml" not in injection["meta"]
    assert result_set["meta"]["instrument_method_xml"]
    acquisition = next(iter(result_set["meta"]["methods"].values()))

    def names(section):
        yield section["name"]
        for child in section["sections"]:
            yield from names(child)

    assert "Oven" in set(names(acquisition))
    assert any(name.endswith(".rst") for name in result_set["meta"]["containers"])


def test_compound_file_rejects_other_files(tmp_path):
    path = tmp_path / "not_ole.dat"
    path.write_bytes(b"plain text, not a container")
    with pytest.raises(ValueError):
        ezchrom.CompoundFile(path)
