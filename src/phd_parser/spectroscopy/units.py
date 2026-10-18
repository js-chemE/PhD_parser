"""The spectral axis in any unit: wavenumber, wavelength, frequency, photon energy, and the Raman shift.

A spectrum is stored once, as a wavenumber in m⁻¹ (``SpectroscopyData``'s
``wavenumber`` coordinate). Every other way of reading the axis is a
conversion of that number, done on demand, so nothing is stored twice and no
processing step has to know which unit the user prefers. This module is that
conversion table; it mirrors the unit model of the IR-bands atlas
(``frontend/src/lib/units.ts``), so a number means the same thing here and
there.

Three kinds of unit, three conversions from a wavenumber ν̃ (m⁻¹):

- a wavenumber scales: ``cm^-1`` is ν̃ / 100
- a wavelength inverts: λ = 1 / ν̃
- a frequency or an energy scales through c and h: f = c ν̃, E = h c ν̃

A **shift** (the Raman convention) is an axis read from a zero, the laser
line, instead of from nothing. In wavenumber, frequency and energy units the
shift is the distance from the zero, which is what "Raman shift / cm^-1"
means and why it needs no laser. In wavelength units nobody plots a
distance: the wavelength axis of a Raman spectrum is the absolute wavelength
of the scattered light, zero − ν̃, and that does need the laser.
"""

from typing import Literal, Union

import numpy as np
import numpy.typing as npt
from scipy import constants as const

Quantity = Literal["wavenumber", "wavelength", "frequency", "energy"]

# unit -> (quantity, factor): value = factor * SI value of that quantity
# (SI: wavenumber m^-1, wavelength m, frequency Hz, energy J)
_UNITS: dict[str, tuple[Quantity, float]] = {
    # wavenumber
    "m^-1": ("wavenumber", 1.0),
    "dm^-1": ("wavenumber", 1e-1),
    "cm^-1": ("wavenumber", 1e-2),
    "mm^-1": ("wavenumber", 1e-3),
    # wavelength
    "m": ("wavelength", 1.0),
    "cm": ("wavelength", 1e2),
    "mm": ("wavelength", 1e3),
    "um": ("wavelength", 1e6),
    "nm": ("wavelength", 1e9),
    "pm": ("wavelength", 1e12),
    "A": ("wavelength", 1e10),
    # frequency
    "Hz": ("frequency", 1.0),
    "kHz": ("frequency", 1e-3),
    "MHz": ("frequency", 1e-6),
    "GHz": ("frequency", 1e-9),
    "THz": ("frequency", 1e-12),
    # energy, per photon
    "J": ("energy", 1.0),
    "meV": ("energy", 1e3 / const.electron_volt),
    "eV": ("energy", 1.0 / const.electron_volt),
    "keV": ("energy", 1e-3 / const.electron_volt),
    "MeV": ("energy", 1e-6 / const.electron_volt),
    # energy, per mole
    "J mol^-1": ("energy", const.Avogadro),
    "kJ mol^-1": ("energy", const.Avogadro * 1e-3),
    "cal mol^-1": ("energy", const.Avogadro / const.calorie),
    "kcal mol^-1": ("energy", const.Avogadro / const.calorie * 1e-3),
}

# Spellings people type, mapped onto the power-notation names above.
_ALIASES: dict[str, str] = {
    "1/m": "m^-1", "m-1": "m^-1", "m⁻¹": "m^-1",
    "1/cm": "cm^-1", "cm-1": "cm^-1", "cm⁻¹": "cm^-1", "wavenumber": "cm^-1", "per_cm": "cm^-1",
    "1/mm": "mm^-1", "mm-1": "mm^-1", "mm⁻¹": "mm^-1",
    "1/dm": "dm^-1", "dm-1": "dm^-1", "dm⁻¹": "dm^-1",
    "µm": "um", "μm": "um", "micrometre": "um", "micrometer": "um",
    "Å": "A", "angstrom": "A", "Angstrom": "A",
    "J/mol": "J mol^-1", "kJ/mol": "kJ mol^-1", "cal/mol": "cal mol^-1", "kcal/mol": "kcal mol^-1",
}

SPECTRAL_UNITS: tuple[str, ...] = tuple(_UNITS)
"""Every unit :func:`to_wavenumber` and :func:`from_wavenumber` accept (aliases such as ``cm-1`` or ``kJ/mol`` work too)."""


def normalise_unit(unit: str) -> str:
    """The canonical spelling of a spectral unit (power notation), e.g. ``'cm-1'`` becomes ``'cm^-1'``.

    Parameters
    ----------
    unit : str
        A unit name or one of its common spellings.

    Returns
    -------
    str
        The key used in :data:`SPECTRAL_UNITS`.

    Raises
    ------
    ValueError
        For a unit this module does not know.
    """
    name = _ALIASES.get(unit.strip(), unit.strip())
    if name not in _UNITS:
        raise ValueError(f"Unknown spectral unit {unit!r}; known units: {', '.join(SPECTRAL_UNITS)}")
    return name


def quantity_of(unit: str) -> Quantity:
    """Which physical quantity a unit measures: ``'wavenumber'``, ``'wavelength'``, ``'frequency'`` or ``'energy'``."""
    return _UNITS[normalise_unit(unit)][0]


def _si_from_wavenumber(wavenumber_si: npt.NDArray, quantity: Quantity) -> npt.NDArray:
    if quantity == "wavenumber":
        return wavenumber_si
    if quantity == "wavelength":
        with np.errstate(divide="ignore"):
            return 1.0 / wavenumber_si
    if quantity == "frequency":
        return wavenumber_si * const.c
    return wavenumber_si * const.Planck * const.c  # energy


def _wavenumber_from_si(value_si: npt.NDArray, quantity: Quantity) -> npt.NDArray:
    if quantity == "wavenumber":
        return value_si
    if quantity == "wavelength":
        with np.errstate(divide="ignore"):
            return 1.0 / value_si
    if quantity == "frequency":
        return value_si / const.c
    return value_si / (const.Planck * const.c)  # energy


def from_wavenumber(
    wavenumber_si: Union[float, npt.NDArray],
    unit: str,
    excitation_wavenumber_si: Union[float, None] = None,
) -> Union[float, npt.NDArray]:
    """Read a stored wavenumber axis (m⁻¹) in another unit.

    Parameters
    ----------
    wavenumber_si : float or numpy.ndarray
        The axis as stored: a wavenumber in m⁻¹, or for Raman data the
        Raman shift in m⁻¹.
    unit : str
        The unit to read it in, any of :data:`SPECTRAL_UNITS` or an alias.
    excitation_wavenumber_si : float or None, optional
        The laser line in m⁻¹ for a shift axis. Needed, and used, only for a
        wavelength unit: the result is then the absolute wavelength of the
        scattered light, ``1 / (excitation - shift)``. For a plain axis leave
        it ``None``.

    Returns
    -------
    float or numpy.ndarray
        The axis in ``unit``; a scalar in gives a scalar out.

    Raises
    ------
    ValueError
        For an unknown unit.
    """
    quantity, factor = _UNITS[normalise_unit(unit)]
    values = np.asarray(wavenumber_si, dtype=float)
    if quantity == "wavelength" and excitation_wavenumber_si is not None:
        values = float(excitation_wavenumber_si) - values  # the scattered light itself
    out = _si_from_wavenumber(values, quantity) * factor
    return float(out) if np.ndim(wavenumber_si) == 0 else out


def to_wavenumber(
    value: Union[float, npt.NDArray],
    unit: str,
    excitation_wavenumber_si: Union[float, None] = None,
) -> Union[float, npt.NDArray]:
    """A value in any spectral unit as the stored wavenumber in m⁻¹ (the inverse of :func:`from_wavenumber`).

    Parameters
    ----------
    value : float or numpy.ndarray
        The value(s) to convert.
    unit : str
        Their unit, any of :data:`SPECTRAL_UNITS` or an alias.
    excitation_wavenumber_si : float or None, optional
        The laser line in m⁻¹ when ``value`` is the absolute wavelength of
        scattered light that should become a Raman shift; ``None`` otherwise.

    Returns
    -------
    float or numpy.ndarray
        Wavenumber (or shift) in m⁻¹; a scalar in gives a scalar out.
    """
    quantity, factor = _UNITS[normalise_unit(unit)]
    values = np.asarray(value, dtype=float) / factor
    out = _wavenumber_from_si(values, quantity)
    if quantity == "wavelength" and excitation_wavenumber_si is not None:
        out = float(excitation_wavenumber_si) - out
    return float(out) if np.ndim(value) == 0 else out


def axis_label(unit: str, shift: bool = False) -> str:
    """The plot label of an axis in ``unit``, in the ``quantity / unit`` form of the house style.

    Parameters
    ----------
    unit : str
        The unit.
    shift : bool, optional
        ``True`` for a Raman shift axis (``raman shift / cm^-1``); a
        wavelength unit of a shift reads the scattered light and is labelled
        so (default is ``False``).

    Returns
    -------
    str
        For example ``'wavenumber / cm^-1'`` or ``'photon energy / eV'``.
    """
    name = normalise_unit(unit)
    quantity = _UNITS[name][0]
    if shift:
        label = "scattered wavelength" if quantity == "wavelength" else "raman shift"
    else:
        label = {"wavenumber": "wavenumber", "wavelength": "wavelength", "frequency": "frequency",
                 "energy": "photon energy"}[quantity]
        if name.endswith("mol^-1"):
            label = "molar photon energy"
    return f"{label} / {name}"
