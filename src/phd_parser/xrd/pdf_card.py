"""Reader for ICDD PDF reference cards exported as tab-separated text.

DIFFRAC.EVA exports the reflection list of a PDF card as a tab-separated
table with one header row::

    d(Å)	2Theta (°)	I var	h	k	l	Comment
    3.51652	25.307	999	1	0	1
    2.43077	36.951	86	1	0	3

The 2θ column of the card is listed for Cu Kα1 (1.5406 Å).
:class:`ReferenceCard` recomputes it from the d-spacings for any other
wavelength.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)

#: Wavelength in Å the 2θ column of an ICDD card is listed for.
CARD_WAVELENGTH_ANGSTROM: float = 1.5406

#: Columns of the reflection table, in this order.
REFLECTION_COLUMNS: list[str] = [
    "d_spacing_angstrom",
    "two_theta_deg",
    "intensity",
    "h",
    "k",
    "l",
    "comment",
]

_CARD_NUMBER = re.compile(r"(\d{2}-\d{3,4})")


def _column_name(raw: str) -> Optional[str]:
    """Map a header cell of the export onto a column of :data:`REFLECTION_COLUMNS`."""
    key = raw.strip().lower()
    if key.startswith("d"):
        return "d_spacing_angstrom"
    if key.startswith("2theta") or key.startswith("2θ") or key.startswith("2 theta"):
        return "two_theta_deg"
    if key.startswith("i"):
        return "intensity"
    if key in ("h", "k", "l", "comment"):
        return key
    return None


def parse_card_name(filename: Union[str, Path]) -> tuple[Optional[str], str]:
    """Card number and phase name from a card file name.

    ``PDF 78-2486_TiO2-Anatase.txt`` gives ``("78-2486", "TiO2-Anatase")``,
    ``MEP_XRD_Reference_PDF 35-0609_MoO3.txt`` gives ``("35-0609", "MoO3")``.

    Parameters
    ----------
    filename : str or Path
        The file name or path of the card.

    Returns
    -------
    tuple of (str or None, str)
        The card number (``None`` when the name holds none) and the
        phase name (the whole stem when there is no card number).
    """
    stem = Path(filename).stem
    match = _CARD_NUMBER.search(stem)
    if match is None:
        return None, stem
    phase = stem[match.end():].strip(" _-")
    return match.group(1), phase or stem


def read_pdf_card(filepath: Union[str, Path]) -> dict[str, Any]:
    """Read an ICDD PDF card exported as tab-separated text.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.txt`` export.

    Returns
    -------
    dict
        ``{"data": DataFrame, "meta": dict}``. The frame has the columns
        ``d_spacing_angstrom``, ``two_theta_deg``, ``intensity`` (floats),
        ``h``, ``k``, ``l`` (ints) and ``comment`` (str, ``""`` when
        empty), one row per reflection in file order. ``meta`` holds
        ``card_number`` and ``phase`` (from the file name),
        ``source_file``, ``n_reflections``, ``intensity_max`` and
        ``intensity_scale`` (999 when the strongest line is above 100,
        else 100).

    Raises
    ------
    ValueError
        When the file lacks the d-spacing, 2θ or intensity column.
    """
    filepath = Path(filepath)
    frame = pd.read_csv(
        filepath,
        sep="\t",
        encoding="utf-8",
        encoding_errors="replace",
        dtype=str,
        keep_default_na=False,
        skip_blank_lines=True,
    )
    frame.columns = [str(column).strip() for column in frame.columns]

    rename: dict[str, str] = {}
    for raw in frame.columns:
        name = _column_name(raw)
        if name is None:
            logger.warning("%s: column %r not recognised, dropped", filepath.name, raw)
        elif name in rename.values():
            logger.warning("%s: column %r repeats %r, dropped", filepath.name, raw, name)
        else:
            rename[raw] = name
    frame = frame[list(rename)].rename(columns=rename)

    missing = {"d_spacing_angstrom", "two_theta_deg", "intensity"} - set(frame.columns)
    if missing:
        raise ValueError(f"{filepath} lacks the column(s) {sorted(missing)}")

    for column in ("d_spacing_angstrom", "two_theta_deg", "intensity"):
        frame[column] = pd.to_numeric(frame[column].str.strip(), errors="coerce").astype(np.float64)
    for column in ("h", "k", "l"):
        if column not in frame.columns:
            frame[column] = 0
        frame[column] = pd.to_numeric(frame[column].str.strip(), errors="coerce").fillna(0).astype(int)
    if "comment" not in frame.columns:
        frame["comment"] = ""
    frame["comment"] = frame["comment"].fillna("").astype(str).str.strip()

    frame = frame.dropna(subset=["d_spacing_angstrom"])[REFLECTION_COLUMNS].reset_index(drop=True)

    card_number, phase = parse_card_name(filepath)
    intensity_max = float(frame["intensity"].max()) if len(frame) else float("nan")
    meta: dict[str, Any] = {
        "card_number": card_number,
        "phase": phase,
        "source_file": str(filepath),
        "n_reflections": int(len(frame)),
        "intensity_max": intensity_max,
        "intensity_scale": 999 if intensity_max > 100 else 100,
    }
    return {"data": frame, "meta": meta}


@dataclass
class ReferenceCard:
    """The reflection list of one ICDD PDF card.

    Parameters
    ----------
    card_number : str or None
        The PDF number, e.g. ``"78-2486"``.
    phase : str
        The phase as named in the file, e.g. ``"TiO2-Anatase"``.
    reflections : pandas.DataFrame
        The table of :func:`read_pdf_card`, columns
        :data:`REFLECTION_COLUMNS`.
    meta : dict, optional
        The ``meta`` of :func:`read_pdf_card`.
    """

    card_number: Optional[str]
    phase: str
    reflections: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> "ReferenceCard":
        """Read a card with :func:`read_pdf_card`.

        Parameters
        ----------
        filepath : str or Path
            Path to the ``.txt`` export.

        Returns
        -------
        ReferenceCard
            The card with its reflections and metadata.
        """
        parsed = read_pdf_card(filepath)
        meta = parsed["meta"]
        return cls(
            card_number=meta["card_number"],
            phase=meta["phase"],
            reflections=parsed["data"],
            meta=meta,
        )

    @property
    def n_reflections(self) -> int:
        """Number of reflections on the card.

        Returns
        -------
        int
            Rows of :attr:`reflections`.
        """
        return int(len(self.reflections))

    @property
    def d_spacing_angstrom(self) -> npt.NDArray[np.float64]:
        """d-spacings in Å, one per reflection.

        Returns
        -------
        ndarray of float
            The ``d_spacing_angstrom`` column.
        """
        return self.reflections["d_spacing_angstrom"].to_numpy(dtype=np.float64)

    @property
    def intensity(self) -> npt.NDArray[np.float64]:
        """Intensities as listed on the card.

        Returns
        -------
        ndarray of float
            The ``intensity`` column, on the card's own scale (999 or 100).
        """
        return self.reflections["intensity"].to_numpy(dtype=np.float64)

    @property
    def relative_intensity(self) -> npt.NDArray[np.float64]:
        """Intensities scaled so the strongest line is 1.

        Returns
        -------
        ndarray of float
            ``intensity / max(intensity)``.
        """
        intensity = self.intensity
        return intensity / intensity.max()

    def two_theta(self, wavelength_angstrom: float = CARD_WAVELENGTH_ANGSTROM) -> npt.NDArray[np.float64]:
        """2θ in degrees of every reflection for a given wavelength.

        Bragg's law from the d-spacings: ``2θ = 2 arcsin(λ / 2d)``. A
        reflection that the wavelength cannot reach (``λ > 2d``) is
        ``nan``.

        Parameters
        ----------
        wavelength_angstrom : float, optional
            X-ray wavelength in Å (default is Cu Kα1, 1.5406 Å, which
            reproduces the card's own 2θ column).

        Returns
        -------
        ndarray of float
            2θ in degrees, one per reflection.
        """
        ratio = wavelength_angstrom / (2.0 * self.d_spacing_angstrom)
        with np.errstate(invalid="ignore"):
            two_theta = np.where(ratio <= 1.0, 2.0 * np.degrees(np.arcsin(np.clip(ratio, 0.0, 1.0))), np.nan)
        return two_theta

    def strongest(self, n: int = 5) -> pd.DataFrame:
        """The ``n`` most intense reflections, strongest first.

        Parameters
        ----------
        n : int, optional
            How many reflections to return (default is 5).

        Returns
        -------
        pandas.DataFrame
            Rows of :attr:`reflections` sorted by descending intensity,
            the original row index kept.
        """
        return self.reflections.sort_values("intensity", ascending=False, kind="stable").head(n)

    def select_two_theta_range(
        self,
        min_deg: Optional[float] = None,
        max_deg: Optional[float] = None,
        wavelength_angstrom: float = CARD_WAVELENGTH_ANGSTROM,
    ) -> pd.DataFrame:
        """The reflections inside a 2θ window, both bounds inclusive.

        Parameters
        ----------
        min_deg : float, optional
            Lower 2θ bound in degrees; open when omitted.
        max_deg : float, optional
            Upper 2θ bound in degrees; open when omitted.
        wavelength_angstrom : float, optional
            Wavelength in Å the window refers to (default is Cu Kα1).

        Returns
        -------
        pandas.DataFrame
            Rows of :attr:`reflections` inside the window, with a
            ``two_theta_deg`` column recomputed for the wavelength.
        """
        two_theta = self.two_theta(wavelength_angstrom)
        mask = ~np.isnan(two_theta)
        if min_deg is not None:
            mask &= two_theta >= min_deg
        if max_deg is not None:
            mask &= two_theta <= max_deg
        selected = self.reflections.loc[mask].copy()
        selected["two_theta_deg"] = two_theta[mask]
        return selected
