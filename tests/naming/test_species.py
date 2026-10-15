"""Species naming conventions: the standard convention, user conventions, and the matching rules."""
import json

import numpy as np
import pandas as pd
import pytest

from phd_parser.naming import SpeciesConvention, get_convention, rename_species
from phd_parser.naming.species import STANDARD_CONVENTION_PATH, normalize_species_key


def test_standard_resolves_the_usual_spellings():
    spellings = ["CH4", "ch4", "Methane", "Methan", "CH₄", " methane "]
    assert rename_species(spellings) == ["CH4"] * 6
    assert rename_species(["Methanol", "MeOH", "CH3OH", "methanol"]) == ["MeOH"] * 4
    assert rename_species(["carbon dioxide", "Kohlendioxid", "CO₂", "co2"]) == ["CO2"] * 4


def test_standard_covers_the_ezchrom_method_names():
    # the names in the Box 2 data analysis method
    vendor = ["CH4", "DME", "Methanol", "MF", "Ethanol", "1-propanol", "isobutanol", "1-butanol",
              "CO2", "H2", "Ar", "N2", "CO"]
    assert rename_species(vendor) == ["CH4", "DME", "MeOH", "MF", "EtOH", "PrOH", "iBuOH", "BuOH",
                                      "CO2", "H2", "Ar", "N2", "CO"]


def test_standard_names_are_those_of_phd_models():
    names = set(SpeciesConvention.standard().canonical_names)
    assert {"CO2", "CO", "H2", "MeOH", "H2O", "CH4", "DME", "MF", "EtOH", "PrOH", "Ar", "N2", "He"} <= names


def test_container_type_is_kept():
    convention = get_convention()
    assert rename_species("Methan") == "CH4"
    assert rename_species(("Methan",)) == ("CH4",)
    assert rename_species(pd.Series(["Methan"])).tolist() == ["CH4"]
    assert list(rename_species(pd.Index(["Methan"]))) == ["CH4"]
    assert rename_species(np.array(["Methan"])).tolist() == ["CH4"]
    assert convention is SpeciesConvention.standard()


def test_unknown_names():
    assert rename_species("unobtainium") == "unobtainium"
    assert rename_species("unobtainium", unknown="none") is None
    with pytest.raises(KeyError):
        rename_species("unobtainium", unknown="raise")


def test_user_convention_from_dict_path_and_flat_form(tmp_path):
    full = {"name": "box2", "species": {"Methane": {"aliases": ["CH4", "C1"]}}}
    assert rename_species("C1", full) == "Methane"
    path = tmp_path / "box2.json"
    path.write_text(json.dumps(full))
    assert rename_species("ch4", path) == "Methane"
    assert rename_species("hydrogen", {"H2": ["H2", "Hydrogen"]}) == "H2"  # phd_ui's flat form


def test_ambiguous_convention_is_refused():
    with pytest.raises(ValueError, match="ambiguous"):
        SpeciesConvention.from_dict({"A": ["x-1"], "B": ["X1"]})


def test_extend_adds_a_setup_spelling_to_the_standard():
    extended = SpeciesConvention.standard().extend({"CH4": ["C1"], "Xe": ["xenon"]})
    assert extended.rename(["C1", "Methan", "xenon"]) == ["CH4", "CH4", "Xe"]
    assert "C1" not in SpeciesConvention.standard()


def test_json_round_trip(tmp_path):
    path = tmp_path / "copy.json"
    SpeciesConvention.standard().to_json(path)
    assert SpeciesConvention.from_json(path).species == SpeciesConvention.standard().species
    assert json.loads(STANDARD_CONVENTION_PATH.read_text(encoding="utf-8"))["name"] == "phd_standard"


def test_matching_key():
    assert normalize_species_key("CO₂") == normalize_species_key(" co-2 ") == "co2"
