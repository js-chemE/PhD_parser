"""Reader for VAMAS files (ISO 14976), the standard exchange format of surface analysis.

A ``.vms`` file is plain text, one value per line, in the order the standard
prescribes: an experiment header, then one block per spectrum (region) with
its acquisition parameters and the ordinate values. Thermo Avantage writes
one for a whole experiment, so one file usually holds every region of a
sample (``C1s Scan``, ``O1s Scan``, ``XPS Survey``, ...). This is what
CasaXPS reads, and it is the instrument's own numbers: counts per channel,
the transmission function next to them, kinetic energy as the axis.

Only the parts of the standard that these files use are implemented: the
``NORM`` experiment mode with ``REGULAR`` scans and an empty parameter
inclusion list. A file using mapping, depth profiling or the inclusion list
raises ``NotImplementedError`` naming the feature rather than reading it
wrong.

``read_vamas`` returns the usual ``{"data", "meta"}`` dict: ``"data"`` maps
each block identifier to its axes and variables, ``"meta"`` holds the
experiment header and, per block, every acquisition parameter.
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Union

import numpy as np

logger = logging.getLogger(__name__)

FORMAT_IDENTIFIER = "VAMAS Surface Chemical Analysis Standard Data Transfer Format 1988 May 4"
MISSING = 1e37  # the standard's "not specified" marker for a number
_SPUTTERING_MODES = ("MAPDP", "MAPSVDP", "SDP", "SDPSV")
_MAP_MODES = ("MAP", "MAPDP", "MAPSV", "MAPSVDP", "SEM")


class _Lines:
    """A cursor over the lines of the file: every field of the standard is one line."""

    def __init__(self, lines: list[str]):
        self.lines = lines
        self.position = 0

    def text(self) -> str:
        """The next line, as text."""
        if self.position >= len(self.lines):
            raise ValueError("Unexpected end of VAMAS file")
        value = self.lines[self.position]
        self.position += 1
        return value

    def number(self) -> float:
        """The next line, as a float."""
        raw = self.text().strip()
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"Expected a number on line {self.position} of the VAMAS file, got {raw!r}") from exc

    def integer(self) -> int:
        """The next line, as an int."""
        return int(self.number())

    def optional(self) -> Union[float, None]:
        """A number, or ``None`` when the file says 'not specified' (1e37)."""
        value = self.number()
        return None if value >= MISSING else value


def _read_header(cursor: _Lines) -> dict[str, Any]:
    header: dict[str, Any] = {}
    identifier = cursor.text().strip()
    if not identifier.startswith("VAMAS Surface Chemical Analysis Standard Data Transfer Format"):
        raise ValueError(f"Not a VAMAS file: first line is {identifier!r}")
    header["format"] = identifier
    header["institution"] = cursor.text().strip()
    header["instrument_model"] = cursor.text().strip()
    header["operator"] = cursor.text().strip()
    header["experiment"] = cursor.text().strip()
    n_comment = cursor.integer()
    header["comment"] = [cursor.text().rstrip("\r") for _ in range(n_comment)]
    header["experiment_mode"] = cursor.text().strip()
    header["scan_mode"] = cursor.text().strip()
    mode = header["experiment_mode"]
    if mode in ("MAP", "MAPDP", "NORM", "SDP"):
        header["n_spectral_regions"] = cursor.integer()
    if mode in ("MAP", "MAPDP"):
        header["n_analysis_positions"] = cursor.integer()
        header["n_discrete_x"] = cursor.integer()
        header["n_discrete_y"] = cursor.integer()
    n_variables = cursor.integer()
    header["experimental_variables"] = [
        {"label": cursor.text().strip(), "unit": cursor.text().strip()} for _ in range(n_variables)
    ]
    n_inclusion = cursor.integer()
    header["parameter_inclusion_list"] = [cursor.integer() for _ in range(abs(n_inclusion))]
    if n_inclusion != 0:
        raise NotImplementedError(
            "This VAMAS file uses a parameter inclusion/exclusion list (blocks that omit fields); not supported."
        )
    n_manual = cursor.integer()
    header["manually_entered_items"] = [cursor.integer() for _ in range(n_manual)]
    n_future_experiment = cursor.integer()
    n_future_block = cursor.integer()
    header["future_upgrade_experiment_entries"] = [cursor.text() for _ in range(n_future_experiment)]
    header["n_future_upgrade_block_entries"] = n_future_block
    header["n_blocks"] = cursor.integer()
    if mode not in ("NORM",):
        raise NotImplementedError(f"VAMAS experiment mode {mode!r} is not supported (only NORM).")
    if header["scan_mode"] != "REGULAR":
        raise NotImplementedError(f"VAMAS scan mode {header['scan_mode']!r} is not supported (only REGULAR).")
    return header


def _read_block(cursor: _Lines, header: dict[str, Any]) -> dict[str, Any]:
    mode = header["experiment_mode"]
    block: dict[str, Any] = {}
    block["identifier"] = cursor.text().strip()
    block["sample"] = cursor.text().strip()
    year, month, day = cursor.integer(), cursor.integer(), cursor.integer()
    hours, minutes, seconds = cursor.integer(), cursor.integer(), cursor.integer()
    gmt_offset_hours = cursor.integer()
    try:
        block["datetime"] = datetime(year, month, day, hours, minutes, seconds,
                                     tzinfo=timezone(timedelta(hours=gmt_offset_hours)))
    except ValueError:
        block["datetime"] = None
    n_comment = cursor.integer()
    block["comment"] = [cursor.text().rstrip("\r") for _ in range(n_comment)]
    technique = cursor.text().strip()
    block["technique"] = technique
    if mode in ("MAP", "MAPDP"):
        block["x_coordinate"] = cursor.number()
        block["y_coordinate"] = cursor.number()
    block["experimental_variable_values"] = [cursor.optional() for _ in header["experimental_variables"]]
    block["source_label"] = cursor.text().strip()
    if mode in _SPUTTERING_MODES:
        block["sputtering_ion_atomic_number"] = cursor.integer()
        block["sputtering_ion_atoms"] = cursor.integer()
        block["sputtering_ion_charge"] = cursor.integer()
    block["source_energy_eV"] = cursor.optional()
    block["source_strength"] = cursor.optional()
    block["source_beam_width_x_um"] = cursor.optional()
    block["source_beam_width_y_um"] = cursor.optional()
    if mode in _MAP_MODES:
        block["field_of_view_x_um"] = cursor.optional()
        block["field_of_view_y_um"] = cursor.optional()
    if mode in ("MAPSV", "MAPSVDP", "SEM"):
        for key in ("first_linescan_start_x", "first_linescan_start_y", "first_linescan_finish_x",
                    "first_linescan_finish_y", "last_linescan_finish_x", "last_linescan_finish_y"):
            block[key] = cursor.optional()
    block["source_polar_angle_deg"] = cursor.optional()
    block["source_azimuth_deg"] = cursor.optional()
    block["analyser_mode"] = cursor.text().strip()
    block["pass_energy_eV"] = cursor.optional()
    if technique == "AES diff":
        block["differential_width_eV"] = cursor.optional()
    block["analyser_magnification"] = cursor.optional()
    block["analyser_work_function_eV"] = cursor.optional()
    block["target_bias_V"] = cursor.optional()
    block["analysis_width_x_um"] = cursor.optional()
    block["analysis_width_y_um"] = cursor.optional()
    block["take_off_polar_angle_deg"] = cursor.optional()
    block["take_off_azimuth_deg"] = cursor.optional()
    block["species_label"] = cursor.text().strip()
    block["transition_label"] = cursor.text().strip()
    block["particle_charge"] = cursor.integer()
    block["abscissa_label"] = cursor.text().strip()
    block["abscissa_unit"] = cursor.text().strip()
    block["abscissa_start"] = cursor.number()
    block["abscissa_increment"] = cursor.number()
    n_variables = cursor.integer()
    block["variables"] = [{"label": cursor.text().strip(), "unit": cursor.text().strip()} for _ in range(n_variables)]
    block["signal_mode"] = cursor.text().strip()
    block["dwell_time_s"] = cursor.optional()
    block["n_scans"] = cursor.integer()
    block["signal_time_correction_s"] = cursor.optional()
    if technique in ("AES diff", "AES dir", "EDX", "ELS", "UPS", "XPS", "XRF") and mode in _SPUTTERING_MODES:
        for key in ("sputtering_energy_eV", "sputtering_beam_current_nA", "sputtering_width_x_um",
                    "sputtering_width_y_um", "sputtering_polar_angle_deg", "sputtering_azimuth_deg"):
            block[key] = cursor.optional()
        block["sputtering_mode"] = cursor.text().strip()
    block["sample_tilt_polar_deg"] = cursor.optional()
    block["sample_tilt_azimuth_deg"] = cursor.optional()
    block["sample_rotation_deg"] = cursor.optional()
    n_additional = cursor.integer()
    block["additional_parameters"] = [
        {"label": cursor.text().strip(), "unit": cursor.text().strip(), "value": cursor.text().strip()}
        for _ in range(n_additional)
    ]
    block["future_upgrade_entries"] = [cursor.text() for _ in range(header["n_future_upgrade_block_entries"])]
    n_ordinate = cursor.integer()
    block["variable_ranges"] = [(cursor.number(), cursor.number()) for _ in block["variables"]]
    n_points, remainder = divmod(n_ordinate, max(n_variables, 1))
    if remainder:
        raise ValueError(
            f"Block {block['identifier']!r}: {n_ordinate} ordinate values do not divide into {n_variables} variables"
        )
    values = np.array([cursor.number() for _ in range(n_ordinate)], dtype=float).reshape(n_points, n_variables)
    block["n_points"] = n_points
    block["values"] = values
    return block


def read_vamas(filepath: Union[str, Path]) -> dict[str, Any]:
    """Read a VAMAS (``.vms``) file into the ``{"data", "meta"}`` dict of the package.

    Parameters
    ----------
    filepath : str or pathlib.Path
        The ``.vms`` file.

    Returns
    -------
    dict
        ``"data"`` maps each block identifier (``"C1s Scan"``, ...) to a dict
        with the abscissa as ``"kinetic_energy"`` or ``"binding_energy"``
        (whichever the file records, in its unit), every corresponding
        variable under its label in snake case (``"counts"``,
        ``"transmission"``), ``"axis_label"``, ``"axis_unit"`` and
        ``"variables"``. ``"meta"`` holds ``"header"`` (the experiment
        header) and ``"blocks"``, the acquisition parameters per block
        identifier (source energy, pass energy, dwell time, scans,
        acquisition time, sample, species label, ...).

    Raises
    ------
    ValueError
        If the file is not a VAMAS file or its block structure is inconsistent.
    NotImplementedError
        For experiment or scan modes other than ``NORM`` / ``REGULAR``.
    """
    path = Path(filepath)
    with open(path, encoding="latin-1") as handle:
        lines = handle.read().splitlines()
    cursor = _Lines(lines)
    header = _read_header(cursor)
    blocks = [_read_block(cursor, header) for _ in range(header["n_blocks"])]
    trailer = cursor.text().strip() if cursor.position < len(lines) else ""
    if trailer != "end of experiment":
        logger.warning("VAMAS file %s does not end with 'end of experiment' (got %r)", path.name, trailer)

    data: dict[str, dict[str, Any]] = {}
    meta_blocks: dict[str, dict[str, Any]] = {}
    for block in blocks:
        identifier = block["identifier"]
        if identifier in data:
            suffix = 2
            while f"{identifier} ({suffix})" in data:
                suffix += 1
            identifier = f"{identifier} ({suffix})"
        axis = block["abscissa_start"] + block["abscissa_increment"] * np.arange(block["n_points"])
        label = block["abscissa_label"].lower()
        axis_key = "binding_energy" if "binding" in label else "kinetic_energy"
        entry: dict[str, Any] = {
            axis_key: axis,
            "axis_label": block["abscissa_label"],
            "axis_unit": block["abscissa_unit"],
            "variables": [variable["label"] for variable in block["variables"]],
        }
        for column, variable in enumerate(block["variables"]):
            entry[variable["label"].strip().lower().replace(" ", "_")] = block["values"][:, column]
        data[identifier] = entry
        meta_blocks[identifier] = {key: value for key, value in block.items() if key != "values"}
        meta_blocks[identifier]["source_file"] = path.name

    return {"data": data, "meta": {"header": header, "blocks": meta_blocks, "source_file": str(path)}}
