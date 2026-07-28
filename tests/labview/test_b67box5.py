import numpy as np
import pytest

from phd_parser.labview import LVData
from phd_parser.labview.b67box5 import read

OLD_HEADER = (
    "timestamp\tReactor T PV\tAnalytic P PV\tVent P PV\t"
    "F1 He PV\tF1 H2 PV\tF1 CO2 PV\tF2 He PV\tF2 H2 PV\tF2 CO2 PV\tF2 Ar PV\tFeed"
)
# 'F1 CO PV' was added to the export in 2026-07.
NEW_HEADER = (
    "timestamp\tReactor T PV\tAnalytic P PV\tVent P PV\t"
    "F1 He PV\tF1 H2 PV\tF1 CO2 PV\tF1 CO PV\tF2 He PV\tF2 H2 PV\tF2 CO2 PV\tF2 Ar PV\tFeed"
)


def write_log(directory, name, header, n_rows=3, minute=30):
    """Write a synthetic tab-separated LabView log with comma decimals."""
    n_channels = header.count("\t")
    lines = [header]
    for i in range(n_rows):
        values = "\t".join(f"{j + i},500" for j in range(n_channels))
        lines.append(f"23-7-2026 16:{minute}:{i:02d}\t{values}")
    path = directory / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_read_old_format(tmp_path):
    path = write_log(tmp_path, "2026-07-23_16-30-00.txt", OLD_HEADER)
    lv = LVData.from_b67_box5_txt(path)

    assert "F1 CO2 PV" in lv.channels
    assert "F1 CO PV" not in lv.channels
    assert lv.n_samples == 3
    assert lv.ds["Reactor T PV"].dtype == np.float64
    np.testing.assert_allclose(lv.ds["Reactor T PV"].values, [0.5, 1.5, 2.5])


def test_read_new_format(tmp_path):
    path = write_log(tmp_path, "2026-07-23_16-30-00.txt", NEW_HEADER)
    lv = LVData.from_b67_box5_txt(path)

    assert "F1 CO PV" in lv.channels
    assert lv.ds["F1 CO PV"].dtype == np.float64
    assert lv.ds["F1 CO PV"].attrs["species"] == "CO"
    assert lv.ds["F1 CO PV"].attrs["controller"] == "F1"
    assert lv.ds["F1 CO PV"].attrs["unit"] == "mL/min"


def test_read_directory_mixing_both_formats(tmp_path):
    write_log(tmp_path, "2026-07-23_16-30-00.txt", OLD_HEADER, minute=30)
    write_log(tmp_path, "2026-07-23_16-40-00.txt", NEW_HEADER, minute=40)

    lv = LVData.from_b67_box5_txt(tmp_path)

    assert lv.n_samples == 6
    assert "F1 CO PV" in lv.channels
    # The old file did not record the channel -> NaN over its rows only.
    assert np.isnan(lv.ds["F1 CO PV"].values).sum() == 3
    assert not np.isnan(lv.ds["Reactor T PV"].values).any()


def test_unknown_channel_is_skipped(tmp_path):
    path = write_log(tmp_path, "2026-07-23_16-30-00.txt", NEW_HEADER + "\tF3 Brand New PV")
    lv = LVData.from_b67_box5_txt(path)

    assert "F3 Brand New PV" not in lv.channels
    assert "F1 CO PV" in lv.channels


def test_unknown_channel_can_be_kept(tmp_path):
    path = write_log(tmp_path, "2026-07-23_16-30-00.txt", NEW_HEADER + "\tF3 Brand New PV")
    lv = LVData.from_b67_box5_txt(path, keep_unknown_channels=True)

    assert "F3 Brand New PV" in lv.channels
    assert lv.ds["F3 Brand New PV"].dtype == np.float64
    assert lv.ds["F3 Brand New PV"].attrs == {}


def test_tos_is_not_treated_as_a_channel(tmp_path):
    path = write_log(tmp_path, "2026-07-23_16-30-00.txt", NEW_HEADER)
    df, channel_meta, _ = read(path)

    assert "tos" not in channel_meta
    assert "timestamp" not in channel_meta
    assert "tos" in df.columns


def test_empty_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .txt"):
        LVData.from_b67_box5_txt(tmp_path)
