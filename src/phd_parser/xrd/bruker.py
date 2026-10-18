"""Reader for the ``.xy`` export of Bruker DIFFRAC.EVA.

The Bruker D8 Advance-ECO (Cu Kα, LynxEye-XE-T detector) exports a
measurement through DIFFRAC.EVA as a text file with one header line of
``Key: "value"`` pairs followed by two whitespace-separated columns,
2θ in degrees and counts::

    'Id: "" Comment: "" Operator: "Lab Manager" Anode: "Cu" Scantype: "Coupled TwoTheta/Theta" TimePerStep: "34.2" X: "9999" Y: "9999" Z: "9999"
    5.0001 2.339
    5.0104 3.099

The header line opens with a single quote in the files seen so far; the
reader accepts it with or without that quote, and a file without any
header line. A third column, when present, is ignored.
"""

import logging
import re
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

#: Kα1 wavelength in Å per anode material.
ANODE_WAVELENGTH_ANGSTROM: dict[str, float] = {
    "Cu": 1.5406,
    "Co": 1.7890,
    "Mo": 0.7093,
    "Cr": 2.2897,
}

#: Header keys of the export and the names they get in ``meta``.
_HEADER_KEY_NAMES: dict[str, str] = {
    "Id": "id",
    "Comment": "comment",
    "Operator": "operator",
    "Anode": "anode",
    "Scantype": "scan_type",
    "TimePerStep": "time_per_step_s",
    "X": "x",
    "Y": "y",
    "Z": "z",
}

_HEADER_PAIR = re.compile(r'(\w+):\s*"([^"]*)"')


def _camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _is_numeric_row(line: str) -> bool:
    parts = line.split()
    if len(parts) < 2:
        return False
    try:
        float(parts[0])
        float(parts[1])
    except ValueError:
        return False
    return True


def parse_header(line: str) -> dict[str, Any]:
    """Parse the ``Key: "value"`` pairs of a DIFFRAC.EVA header line.

    Parameters
    ----------
    line : str
        The header line, with or without its leading single quote.

    Returns
    -------
    dict
        The pairs under their snake_case names (``id``, ``comment``,
        ``operator``, ``anode``, ``scan_type``, ``time_per_step_s`` as a
        float, ``x``, ``y``, ``z``); a key that is not in the known set
        is kept under its camel-to-snake-cased name. Every known key is
        present, ``None`` when the header does not carry it.
    """
    meta: dict[str, Any] = {name: None for name in _HEADER_KEY_NAMES.values()}
    for key, value in _HEADER_PAIR.findall(line):
        name = _HEADER_KEY_NAMES.get(key, _camel_to_snake(key))
        if name == "time_per_step_s":
            try:
                meta[name] = float(value)
            except ValueError:
                logger.warning("TimePerStep %r is not a number, kept as text", value)
                meta[name] = value
        else:
            meta[name] = value
    return meta


def wavelength_for_anode(anode: Optional[str]) -> Optional[float]:
    """Kα1 wavelength in Å of an anode material, ``None`` when unknown.

    Parameters
    ----------
    anode : str or None
        Anode material as written in the header (``"Cu"``, ``"Co"``,
        ``"Mo"``, ``"Cr"``); case-insensitive.

    Returns
    -------
    float or None
        The Kα1 wavelength in Å, or ``None`` (with a warning) for an
        anode not in :data:`ANODE_WAVELENGTH_ANGSTROM` or no anode.
    """
    if not anode:
        logger.warning("No anode in the header, wavelength left unset")
        return None
    for name, wavelength in ANODE_WAVELENGTH_ANGSTROM.items():
        if name.lower() == anode.strip().lower():
            return wavelength
    logger.warning("Unknown anode %r, wavelength left unset", anode)
    return None


def read_xy(filepath: Union[str, Path]) -> dict[str, Any]:
    """Read a Bruker DIFFRAC.EVA ``.xy`` export.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.xy`` file.

    Returns
    -------
    dict
        ``{"data": {"angle": ndarray, "values": ndarray}, "meta": dict}``.
        ``angle`` is 2θ in degrees in file order, ``values`` the counts.
        ``meta`` holds the parsed header pairs (see :func:`parse_header`),
        the raw header line under ``header`` (``None`` without one),
        ``source_file``, ``n_points``, ``step_deg`` (median step),
        ``angle_range_deg`` (``[min, max]``) and ``wavelength_angstrom``
        from the anode (``None`` for an unknown anode).

    Raises
    ------
    ValueError
        When the file holds no numeric rows.
    """
    filepath = Path(filepath)
    with filepath.open("r", encoding="utf-8", errors="replace") as handle:
        lines = [line.strip() for line in handle]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError(f"{filepath} is empty")

    header: Optional[str] = None
    first_data = 0
    if not _is_numeric_row(lines[0]):
        header = lines[0].lstrip("'").strip()
        first_data = 1
        logger.debug("Header of %s: %s", filepath.name, header)
    else:
        logger.debug("%s has no header line", filepath.name)

    data_lines = [line for line in lines[first_data:] if _is_numeric_row(line)]
    if len(data_lines) < len(lines) - first_data:
        logger.warning(
            "%s: %d non-numeric rows skipped",
            filepath.name,
            len(lines) - first_data - len(data_lines),
        )
    if not data_lines:
        raise ValueError(f"{filepath} holds no numeric rows")

    columns = np.loadtxt(data_lines, usecols=(0, 1), ndmin=2, dtype=np.float64)
    angle = columns[:, 0]
    values = columns[:, 1]

    meta: dict[str, Any] = parse_header(header or "")
    meta["header"] = header
    meta["source_file"] = str(filepath)
    meta["n_points"] = int(angle.size)
    meta["step_deg"] = float(np.median(np.diff(angle))) if angle.size > 1 else None
    meta["angle_range_deg"] = [float(angle.min()), float(angle.max())]
    meta["wavelength_angstrom"] = wavelength_for_anode(meta.get("anode"))

    return {"data": {"angle": angle, "values": values}, "meta": meta}
