"""The Bruker DIFFRAC.EVA ``.xy`` reader and ``XRDData.from_bruker_xy``."""
import logging
from pathlib import Path

import numpy as np
import pytest

from phd_parser.xrd import XRDData, read_xy
from phd_parser.xrd.bruker import ANODE_WAVELENGTH_ANGSTROM, parse_header, wavelength_for_anode
from phd_parser.xrd.xrd_e1290 import read_xy_e1290

HERE = Path(__file__).parent
FIXTURE = HERE / "bruker_SK0003_MoO3-12pct-TiO2_12-28deg.xy"
FULL_FILE = HERE / "xrd_mock_e1290.xy"  # a complete export of the same instrument
HEADER = (
    '\'Id: "" Comment: "" Operator: "Lab Manager" Anode: "Cu" Scantype: "Coupled TwoTheta/Theta" '
    'TimePerStep: "34.2" X: "9999" Y: "9999" Z: "9999" '
)


def test_header_is_parsed_into_meta():
    meta = read_xy(FIXTURE)["meta"]
    assert meta["operator"] == "Lab Manager"
    assert meta["anode"] == "Cu"
    assert meta["scan_type"] == "Coupled TwoTheta/Theta"
    assert meta["time_per_step_s"] == pytest.approx(34.2)
    assert meta["id"] == "" and meta["comment"] == ""
    assert meta["x"] == meta["y"] == meta["z"] == "9999"
    assert meta["header"].startswith('Id: ""') and "'" not in meta["header"]
    assert meta["source_file"] == str(FIXTURE)


def test_points_step_range_and_wavelength():
    parsed = read_xy(FIXTURE)
    angle, values = parsed["data"]["angle"], parsed["data"]["values"]
    meta = parsed["meta"]
    assert angle.shape == values.shape == (1552,)
    assert meta["n_points"] == 1552
    assert meta["step_deg"] == pytest.approx(0.0103, abs=1e-4)
    assert meta["angle_range_deg"] == pytest.approx([12.0028, 27.9985])
    assert meta["wavelength_angstrom"] == pytest.approx(1.5406)
    assert np.all(np.diff(angle) > 0)
    assert values.dtype == np.float64 and angle.dtype == np.float64


def test_same_arrays_as_the_e1290_reader_on_a_full_export():
    parsed = read_xy(FULL_FILE)
    legacy = read_xy_e1290(FULL_FILE, normalize=False)
    assert parsed["meta"]["n_points"] == 8243
    np.testing.assert_allclose(parsed["data"]["angle"], legacy[0])
    np.testing.assert_allclose(parsed["data"]["values"], legacy[1])


def test_from_bruker_xy_sets_metadata_and_attrs():
    xrd = XRDData.from_bruker_xy(FIXTURE)
    assert isinstance(xrd, XRDData)
    assert xrd.ndim == 1
    assert xrd.angle.size == 1552
    assert np.all(np.diff(xrd.angle) > 0)
    assert xrd.da.attrs["wavelength_angstrom"] == pytest.approx(1.5406)
    assert xrd.da.attrs["anode"] == "Cu"
    assert xrd.da.attrs["source_file"] == str(FIXTURE)
    assert xrd.da.attrs["angle_units"] == "deg"
    assert xrd.metadata["scan_type"] == "Coupled TwoTheta/Theta"
    assert xrd.metadata["n_points"] == 1552


def test_from_bruker_xy_normalize():
    xrd = XRDData.from_bruker_xy(FIXTURE, normalize=True)
    assert xrd.values.max() == pytest.approx(1.0)
    raw = XRDData.from_bruker_xy(FIXTURE)
    np.testing.assert_allclose(xrd.values, raw.values / raw.values.max())


def test_file_without_header(tmp_path):
    path = tmp_path / "no_header.xy"
    path.write_text("5.0001 2.339\n5.0104 3.099\n5.0207 3.099\n")
    parsed = read_xy(path)
    assert parsed["meta"]["header"] is None
    assert parsed["meta"]["anode"] is None
    assert parsed["meta"]["wavelength_angstrom"] is None
    np.testing.assert_allclose(parsed["data"]["angle"], [5.0001, 5.0104, 5.0207])
    np.testing.assert_allclose(parsed["data"]["values"], [2.339, 3.099, 3.099])


def test_unquoted_header_and_three_columns(tmp_path):
    path = tmp_path / "variant.xy"
    path.write_text(HEADER.lstrip("'") + "\n5.0001 2.339 1\n5.0104 3.099 1\n5.0207 3.099 2\n")
    parsed = read_xy(path)
    assert parsed["meta"]["anode"] == "Cu"
    assert parsed["meta"]["time_per_step_s"] == pytest.approx(34.2)
    assert parsed["data"]["angle"].shape == (3,)
    np.testing.assert_allclose(parsed["data"]["values"], [2.339, 3.099, 3.099])


def test_unknown_anode_warns_and_leaves_wavelength_unset(tmp_path, caplog):
    path = tmp_path / "ag.xy"
    path.write_text(HEADER.replace('Anode: "Cu"', 'Anode: "Ag"') + "\n5.0 1.0\n5.1 2.0\n")
    with caplog.at_level(logging.WARNING, logger="phd_parser.xrd.bruker"):
        parsed = read_xy(path)
    assert parsed["meta"]["anode"] == "Ag"
    assert parsed["meta"]["wavelength_angstrom"] is None
    assert any("Ag" in record.message for record in caplog.records)


def test_anode_table():
    assert set(ANODE_WAVELENGTH_ANGSTROM) == {"Cu", "Co", "Mo", "Cr"}
    assert wavelength_for_anode("cu") == pytest.approx(1.5406)
    assert wavelength_for_anode("Mo") == pytest.approx(0.7093)


def test_parse_header_keeps_unknown_keys():
    meta = parse_header('Anode: "Co" SampleName: "abc"')
    assert meta["anode"] == "Co"
    assert meta["sample_name"] == "abc"
    assert meta["operator"] is None


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.xy"
    path.write_text("")
    with pytest.raises(ValueError):
        read_xy(path)
    path.write_text(HEADER + "\n")
    with pytest.raises(ValueError):
        read_xy(path)
