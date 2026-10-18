"""Reader for Thermo Avantage ``.avg`` files: the text dump of one DataSpace (one XPS region).

Avantage writes a ``.avg`` next to every ``.VGD`` (its binary store). The
text file is a dump of the DataSpace: ``$PROPERTIES`` sections of
``NAME : VT_TYPE = value`` lines (instrument, source energy, pass energy,
dwell time, scans, start and end time, stage position, lens mode,
transmission function coefficients), ``$SPACEAXES`` describing the energy
axis (start, step, number of points, unit) and ``$DATA`` holding the values
as ``LIST@ i= a, b, c, d`` rows. One file holds one region; a sample's
folder has one per region.

Mind ``DS_ANPROPID_TXFN_REMOVED``: when it is ``True`` the values are
already divided by the analyser transmission function, so they are not the
raw counts of the VAMAS export of the same scan.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import numpy as np

logger = logging.getLogger(__name__)

_PROPERTY = re.compile(r"^(?P<name>[A-Z0-9_\[\]]+)\s*:\s*(?P<type>VT_[A-Z0-9]+)\s*=\s*(?P<value>.*)$")
_SPACE_AXIS = re.compile(r"^\s*(?P<index>\d+)=\s*(?P<fields>.+)$")


def _convert(kind: str, raw: str) -> Any:
    raw = raw.strip()
    if kind == "VT_BSTR":
        return raw.strip("'")
    if kind == "VT_BOOL":
        return raw.lower() == "true"
    if kind in ("VT_I2", "VT_I4", "VT_UI2", "VT_UI4"):
        return int(float(raw))
    if kind in ("VT_R4", "VT_R8"):
        return float(raw)
    if kind == "VT_DATE":
        for pattern in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
            try:
                return datetime.strptime(" ".join(raw.split()), pattern)
            except ValueError:
                continue
        return raw
    return raw


def read_avg(filepath: Union[str, Path]) -> dict[str, Any]:
    """Read a Thermo Avantage ``.avg`` file into the ``{"data", "meta"}`` dict of the package.

    Parameters
    ----------
    filepath : str or pathlib.Path
        The ``.avg`` file.

    Returns
    -------
    dict
        ``"data"`` holds the axis under ``"kinetic_energy"`` (eV, what the
        instrument scans), the values under ``"values"``, and ``"n_points"``.
        ``"meta"`` holds every property as ``{name: value}`` with the
        ``DS_`` prefix kept, the space axis description under ``"axis"``,
        and the derived ``"title"``, ``"source_energy_eV"``,
        ``"pass_energy_eV"``, ``"dwell_time_s"``, ``"n_scans"``,
        ``"start_time"``, ``"end_time"``, ``"value_label"``,
        ``"transmission_removed"`` and ``"source_file"``.

    Raises
    ------
    ValueError
        If the file has no ``$SPACEAXES`` or ``$DATA`` section, or the data
        length does not match the axis.
    """
    path = Path(filepath)
    with open(path, encoding="latin-1") as handle:
        lines = handle.read().splitlines()

    properties: dict[str, Any] = {}
    axis: dict[str, Any] = {}
    values: list[float] = []
    section = None
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        if stripped.startswith("$"):
            section = stripped.split("=", 1)[0]
            continue
        if section == "$PROPERTIES":
            match = _PROPERTY.match(stripped)
            if match:
                properties[match["name"]] = _convert(match["type"], match["value"])
        elif section == "$SPACEAXES":
            match = _SPACE_AXIS.match(stripped)
            if match and not axis:
                fields = [field.strip().strip("'") for field in match["fields"].split(",")]
                axis = {
                    "start": float(fields[0]), "step": float(fields[1]), "n_points": int(fields[2]),
                    "type": fields[3], "scaling": fields[4], "symbol": fields[5], "unit": fields[6],
                    "label": fields[7] if len(fields) > 7 else "",
                }
        elif section == "$DATA":
            if stripped.startswith("LIST@"):
                _, _, numbers = stripped.partition("=")
                values.extend(float(item) for item in numbers.split(",") if item.strip())
    if not axis:
        raise ValueError(f"{path.name}: no $SPACEAXES section found")
    if not values:
        raise ValueError(f"{path.name}: no $DATA section found")
    if len(values) != axis["n_points"]:
        raise ValueError(f"{path.name}: {len(values)} values for an axis of {axis['n_points']} points")

    energy = axis["start"] + axis["step"] * np.arange(axis["n_points"])
    meta = {
        **properties,
        "axis": axis,
        "title": properties.get("DS_EXT_SUPROPID_TITLE", path.stem),
        "sample": properties.get("DS_EXT_SUPROPID_SUBJECT", ""),
        "source_energy_eV": properties.get("DS_SOPROPID_ENERGY"),
        "pass_energy_eV": properties.get("DS_ANPROPID_PASS"),
        "dwell_time_s": properties.get("DS_ACPROPID_ACQ_TIME"),
        "n_scans": properties.get("DS_ACPROPID_PERIODS"),
        "start_time": properties.get("DS_ACPROPID_START_TIME"),
        "end_time": properties.get("DS_ACPROPID_END_TIME"),
        "value_label": properties.get("DS_GEPROPID_VALUE_LABEL", ""),
        "transmission_removed": bool(properties.get("DS_ANPROPID_TXFN_REMOVED", False)),
        "source_file": str(path),
    }
    return {"data": {"kinetic_energy": energy, "values": np.asarray(values, dtype=float), "n_points": axis["n_points"]},
            "meta": meta}
