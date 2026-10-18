"""The ICDD PDF card reader and ``ReferenceCard``."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phd_parser.xrd import ReferenceCard, read_pdf_card
from phd_parser.xrd.pdf_card import REFLECTION_COLUMNS, parse_card_name

HERE = Path(__file__).parent
ANATASE = HERE / "PDF 78-2486_TiO2-Anatase.txt"
MOO3 = HERE / "PDF 35-0609_MoO3.txt"


def test_columns_and_types():
    parsed = read_pdf_card(ANATASE)
    frame = parsed["data"]
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == REFLECTION_COLUMNS
    assert len(frame) == 20
    assert frame["d_spacing_angstrom"].dtype == np.float64
    assert frame["two_theta_deg"].dtype == np.float64
    assert frame["intensity"].dtype == np.float64
    for column in ("h", "k", "l"):
        assert np.issubdtype(frame[column].dtype, np.integer)
    first = frame.iloc[0]
    assert first["d_spacing_angstrom"] == pytest.approx(3.51652)
    assert first["two_theta_deg"] == pytest.approx(25.307)
    assert first["intensity"] == 999
    assert (first["h"], first["k"], first["l"]) == (1, 0, 1)
    assert first["comment"] == ""


def test_meta_from_file_name_and_table():
    meta = read_pdf_card(ANATASE)["meta"]
    assert meta["card_number"] == "78-2486"
    assert meta["phase"] == "TiO2-Anatase"
    assert meta["source_file"] == str(ANATASE)
    assert meta["n_reflections"] == 20
    assert meta["intensity_scale"] == 999
    assert meta["intensity_max"] == 999

    meta = read_pdf_card(MOO3)["meta"]
    assert meta["card_number"] == "35-0609"
    assert meta["phase"] == "MoO3"
    assert meta["n_reflections"] == 58
    assert meta["intensity_scale"] == 100


def test_comment_column_is_kept():
    frame = read_pdf_card(MOO3)["data"]
    strongest = frame.loc[frame["intensity"].idxmax()]
    assert strongest["comment"] == "Strongest line."
    assert strongest["two_theta_deg"] == pytest.approx(27.34)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("PDF 78-2486_TiO2-Anatase.txt", ("78-2486", "TiO2-Anatase")),
        ("MEP_XRD_Reference_PDF 35-0609_MoO3.txt", ("35-0609", "MoO3")),
        ("PDF 70-1707 (NH4)6(Mo7O24)(H2O)4.txt", ("70-1707", "(NH4)6(Mo7O24)(H2O)4")),
        ("anatase.txt", (None, "anatase")),
    ],
)
def test_parse_card_name(name, expected):
    assert parse_card_name(name) == expected


def test_reference_card_from_file():
    card = ReferenceCard.from_file(ANATASE)
    assert card.card_number == "78-2486"
    assert card.phase == "TiO2-Anatase"
    assert card.n_reflections == 20
    assert card.relative_intensity.max() == pytest.approx(1.0)


def test_strongest():
    card = ReferenceCard.from_file(ANATASE)
    top = card.strongest(3)
    assert len(top) == 3
    assert list(top["intensity"]) == [999, 452, 321]
    assert list(zip(top["h"], top["k"], top["l"])) == [(1, 0, 1), (2, 0, 0), (2, 1, 1)]
    assert top["intensity"].is_monotonic_decreasing


def test_two_theta_reproduces_the_card_for_cu_kalpha1():
    for path in (ANATASE, MOO3):
        card = ReferenceCard.from_file(path)
        recomputed = card.two_theta(1.5406)
        np.testing.assert_allclose(recomputed, card.reflections["two_theta_deg"], atol=0.01)


def test_two_theta_for_another_wavelength_follows_bragg():
    card = ReferenceCard.from_file(ANATASE)
    wavelength = 1.7890  # Co Kα1
    expected = 2.0 * np.degrees(np.arcsin(wavelength / (2.0 * card.d_spacing_angstrom)))
    np.testing.assert_allclose(card.two_theta(wavelength), expected)
    assert card.two_theta(wavelength)[0] > card.two_theta(1.5406)[0]
    # a wavelength longer than 2d cannot reach the reflection
    assert np.isnan(card.two_theta(10.0)).all()


def test_select_two_theta_range():
    card = ReferenceCard.from_file(MOO3)
    window = card.select_two_theta_range(12.0, 28.0)
    assert len(window) == 5
    assert window["two_theta_deg"].between(12.0, 28.0).all()
    assert list(window["intensity"]) == [17, 66, 36, 30, 100]
