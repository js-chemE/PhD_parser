"""Species naming conventions: map the many ways a setup writes a species onto one canonical name.

A convention is a human-readable JSON file::

    {
      "name": "my_setup",
      "description": "optional",
      "species": {
        "CH4": {"name": "methane", "formula": "CH4", "aliases": ["Methan", "C1"]},
        "MeOH": {"name": "methanol", "formula": "CH3OH", "aliases": []}
      }
    }

The key is the canonical name. The key, ``name``, ``formula`` and every
alias all resolve to it. The flat form ``{"CH4": ["Methan", "methane"]}``
(canonical name to a list of aliases, as in ``phd_ui``) is accepted too.

Matching ignores case, spaces, hyphens, underscores and dots, and reads
Unicode subscript and superscript digits as digits. Two species claiming the
same alias under that matching is an error when the convention is loaded,
not a silent first-come-first-served lookup later.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

logger = logging.getLogger(__name__)

STANDARD_CONVENTION_PATH = Path(__file__).with_name("naming_species_standard.json")

_IGNORED_CHARACTERS = re.compile(r"[\s\-_.]+")

ConventionLike = Union[None, str, Path, dict, "SpeciesConvention"]
NamesLike = Union[str, Sequence[str], np.ndarray, pd.Series, pd.Index]
Unknown = Literal["keep", "raise", "none"]


def normalize_species_key(name: str) -> str:
    """Return the matching key of a species name.

    Parameters
    ----------
    name : str
        Any spelling, e.g. ``"CO₂"``, ``" carbon-dioxide "``.

    Returns
    -------
    str
        Case-folded, without spaces, hyphens, underscores and dots, subscript
        digits as digits (``"co2"``, ``"carbondioxide"``).
    """
    text = unicodedata.normalize("NFKC", str(name))  # NFKC turns subscript/superscript digits into digits
    return _IGNORED_CHARACTERS.sub("", text).casefold()


class SpeciesEntry(BaseModel):
    """One species of a convention: its full name, formula and alternative spellings."""

    name: str = ""
    formula: str = ""
    aliases: list[str] = Field(default_factory=list)


class SpeciesConvention(BaseModel):
    """A species naming convention with a precomputed lookup.

    Build one with :meth:`from_json`, :meth:`from_dict` or :meth:`standard`.

    Raises
    ------
    ValueError
        If two species share an alias under the matching rules.
    """

    model_config = ConfigDict(frozen=True)

    name: str = "custom"
    description: str = ""
    species: dict[str, SpeciesEntry]

    _lookup: dict[str, str] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: Any) -> None:
        lookup: dict[str, str] = {}
        conflicts: list[str] = []
        for canonical, entry in self.species.items():
            for spelling in {canonical, entry.name, entry.formula, *entry.aliases}:
                if not spelling:
                    continue
                key = normalize_species_key(spelling)
                if lookup.get(key, canonical) != canonical:
                    conflicts.append(f"'{spelling}' -> {lookup[key]} and {canonical}")
                lookup[key] = canonical
        if conflicts:
            raise ValueError(f"Convention '{self.name}' is ambiguous: " + "; ".join(conflicts))
        self._lookup = lookup

    # ----------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any], name: Optional[str] = None) -> "SpeciesConvention":
        """Build a convention from an already read JSON object.

        Parameters
        ----------
        data : dict
            Either the full form (with a ``"species"`` key) or the flat form
            ``{canonical: [aliases]}``.
        name : str, optional
            Overrides the ``"name"`` in ``data``.

        Returns
        -------
        SpeciesConvention
            The convention.
        """
        if "species" in data and isinstance(data["species"], dict):
            species = data["species"]
            label, description = data.get("name", "custom"), data.get("description", "")
        else:
            species, label, description = data, "custom", ""
        entries = {
            canonical: SpeciesEntry(aliases=list(value)) if isinstance(value, (list, tuple)) else SpeciesEntry(**value)
            for canonical, value in species.items()
        }
        return cls(name=name or label, description=description, species=entries)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "SpeciesConvention":
        """Read a convention from a JSON file.

        Parameters
        ----------
        path : str or Path
            The JSON file, full or flat form.

        Returns
        -------
        SpeciesConvention
            The convention.
        """
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def standard(cls) -> "SpeciesConvention":
        """Return the standard convention of the phd_* packages (``naming/species/naming_species_standard.json``)."""
        return _standard()

    def to_json(self, path: Union[str, Path]) -> None:
        """Write the convention as human-readable JSON.

        Parameters
        ----------
        path : str or Path
            Destination file.
        """
        data = {"name": self.name, "description": self.description,
                "species": {canonical: entry.model_dump() for canonical, entry in self.species.items()}}
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def extend(self, other: Union[dict, "SpeciesConvention"], name: Optional[str] = None) -> "SpeciesConvention":
        """Return a new convention with another one's species added.

        For a species in both, the aliases are combined and ``other``'s name
        and formula win where set. Use it to add a setup's own spellings on
        top of the standard.

        Parameters
        ----------
        other : dict or SpeciesConvention
            Species to add, in any form :func:`get_convention` accepts.
        name : str, optional
            Name of the result (default: ``"<self>+<other>"``).

        Returns
        -------
        SpeciesConvention
            The combined convention.
        """
        other = get_convention(other)
        species = {canonical: entry.model_copy(deep=True) for canonical, entry in self.species.items()}
        for canonical, entry in other.species.items():
            if canonical in species:
                base = species[canonical]
                species[canonical] = SpeciesEntry(
                    name=entry.name or base.name, formula=entry.formula or base.formula,
                    aliases=list(dict.fromkeys([*base.aliases, *entry.aliases])),
                )
            else:
                species[canonical] = entry.model_copy(deep=True)
        return SpeciesConvention(name=name or f"{self.name}+{other.name}",
                                 description=self.description, species=species)

    # ----------------------------------------------------------------
    # Lookup
    # ----------------------------------------------------------------

    @property
    def canonical_names(self) -> list[str]:
        """The canonical species names, in file order."""
        return list(self.species)

    def __contains__(self, name: str) -> bool:
        return normalize_species_key(name) in self._lookup

    def get_canonical(self, name: str, unknown: Unknown = "keep") -> Optional[str]:
        """Return the canonical name of one spelling.

        Parameters
        ----------
        name : str
            Any spelling.
        unknown : {"keep", "raise", "none"}, optional
            What to return for a name the convention does not know: the name
            unchanged (default, with a warning), raise, or ``None``.

        Returns
        -------
        str or None
            The canonical name.

        Raises
        ------
        KeyError
            If ``unknown="raise"`` and the name is unknown.
        """
        canonical = self._lookup.get(normalize_species_key(name))
        if canonical is not None:
            return canonical
        if unknown == "raise":
            raise KeyError(f"Species '{name}' is not in convention '{self.name}'.")
        if unknown == "none":
            return None
        _warn_unknown(name, self.name)
        return name

    def get_entry(self, name: str) -> SpeciesEntry:
        """Return the entry (full name, formula, aliases) of any spelling.

        Raises
        ------
        KeyError
            If the name is unknown.
        """
        return self.species[self.get_canonical(name, unknown="raise")]

    def rename(self, names: NamesLike, unknown: Unknown = "keep") -> NamesLike:
        """Rename species to their canonical names, keeping the container type.

        Parameters
        ----------
        names : str, sequence of str, numpy.ndarray, pandas.Series or pandas.Index
            Names to rename.
        unknown : {"keep", "raise", "none"}, optional
            See :meth:`get_canonical`.

        Returns
        -------
        same type as ``names``
            The canonical names.
        """
        if isinstance(names, str):
            return self.get_canonical(names, unknown)
        if isinstance(names, pd.Index):
            return names.map(lambda name: self.get_canonical(name, unknown))
        if isinstance(names, pd.Series):
            return names.map(lambda name: self.get_canonical(name, unknown))
        if isinstance(names, np.ndarray):
            return np.array([self.get_canonical(name, unknown) for name in names.tolist()], dtype=object)
        return type(names)(self.get_canonical(name, unknown) for name in names)

    def __repr__(self) -> str:
        return f"SpeciesConvention(name={self.name!r}, species={len(self.species)}, spellings={len(self._lookup)})"

    def __str__(self) -> str:
        return self.__repr__()


@lru_cache(maxsize=1)
def _standard() -> SpeciesConvention:
    return SpeciesConvention.from_json(STANDARD_CONVENTION_PATH)


_warned: set[tuple[str, str]] = set()


def _warn_unknown(name: str, convention: str) -> None:
    if (name, convention) not in _warned:
        _warned.add((name, convention))
        logger.warning(f"Species '{name}' is not in convention '{convention}'; kept as is.")


def get_convention(convention: ConventionLike = None) -> SpeciesConvention:
    """Resolve anything that describes a convention into a :class:`SpeciesConvention`.

    Parameters
    ----------
    convention : None, str, Path, dict or SpeciesConvention, optional
        ``None`` for the standard convention, a path to a JSON file, an
        already read JSON object, or a convention.

    Returns
    -------
    SpeciesConvention
        The convention.

    Raises
    ------
    TypeError
        For anything else.
    """
    if convention is None:
        return _standard()
    if isinstance(convention, SpeciesConvention):
        return convention
    if isinstance(convention, (str, Path)):
        return SpeciesConvention.from_json(convention)
    if isinstance(convention, dict):
        return SpeciesConvention.from_dict(convention)
    raise TypeError(f"Cannot build a species convention from {type(convention)}")


def rename_species(names: NamesLike, convention: ConventionLike = None, unknown: Unknown = "keep") -> NamesLike:
    """Rename species to the canonical names of a convention.

    Parameters
    ----------
    names : str, sequence of str, numpy.ndarray, pandas.Series or pandas.Index
        Names as a setup or software wrote them.
    convention : None, str, Path, dict or SpeciesConvention, optional
        See :func:`get_convention`; ``None`` (default) is the standard.
    unknown : {"keep", "raise", "none"}, optional
        What to do with names the convention does not know (default: keep
        them unchanged and warn once per name).

    Returns
    -------
    same type as ``names``
        The canonical names.

    Examples
    --------
    >>> rename_species(["Methan", "ch4", "CO₂", "Methanol"])
    ['CH4', 'CH4', 'CO2', 'MeOH']
    """
    return get_convention(convention).rename(names, unknown)
