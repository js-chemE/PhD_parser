"""Naming conventions: map the many spellings setups use onto one canonical name (species for now)."""

import phd_parser.naming.species as species
from phd_parser.naming.species import SpeciesConvention, get_convention, rename_species

__all__ = ["species", "SpeciesConvention", "get_convention", "rename_species"]
