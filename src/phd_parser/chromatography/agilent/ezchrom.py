"""Reader for Agilent OpenLab CDS EZChrom Edition result sets (``.rslt`` folders).

Reads the vendor files directly, without OpenLab or any Agilent library. None
of the files are encrypted; the binary ones are OLE2 / Compound File Binary
containers (the container format of old ``.doc``/``.xls`` files).

What an ``.rslt`` folder holds
------------------------------
``*.dat``
    One per injection, OLE2. Raw detector traces as integer ADC counts, the
    full Agilent 7890 instrument method as XML, instrument setup text, the
    instrument configuration, a SQLite audit trail and the integration result.
``*.acaml``, ``*.rst.0.NNNN.acam_``
    ACAML 1.5 XML, Agilent's documented result schema. The ``.acaml`` covers
    the whole sequence, each ``.acam_`` one injection: peaks with every
    integrator field, species amounts, calibration curves with their full
    calibration history, column and liner information.
``*_ACQ_*.macaml``, ``*_DA_*.macaml``
    ACAML method configuration: acquisition method (oven, inlet, flows, valve
    timetable, signals) and data analysis method (peak tables, integration
    events), as nested sections of name / value / unit parameters.
``*.rst``, ``Instrument*.cfg``
    OLE2 containers with the sequence header and the instrument configuration.

Trace encoding
--------------
Verified against the AIA netCDF export of the same injections (120 traces,
largest relative deviation 5.8e-8, i.e. float32 rounding of the export):

- stream ``Detector Data/Detector <id> Trace``: ``int32 x5`` header
  ``(version, n, n + 1, 1, 0)``, then ``n + 1`` float64 values from byte 20 on,
  then a 40 byte trailer with ``float32`` sampling interval in s at +16 and
  ``float32`` scale factor at +20.
- the values are whole numbers, the raw ADC counts. Sample ``k`` is at
  ``t = k * sampling_interval``, from 0 to the run time. The last sample is an
  unacquired zero placeholder at exactly the run time; it is kept, the netCDF
  export drops it.
- signal in detector units = counts * scale factor. The scale factor is
  1/7680, the number ACAML calls ``YAxisScalingFactor``. Units come from the
  ``Detector Trace Handler`` stream (FID ``pA``, TCD ``25 µV``).
- ACAML peak ``Height``, ``Area``, ``LevelStart``, ``LevelEnd`` are in counts
  and counts s, not in detector units. ``RetentionTime``, ``BeginTime``,
  ``EndTime`` are in minutes.

All parsers return plain dicts with ``"data"`` and ``"meta"`` keys.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import struct
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ACAML_NAMESPACE = {"a": "urn:schemas-agilent-com:acaml15"}
_SPECIAL_SECTOR = 0xFFFFFFFA  # sector ids from here up mark end of chain, free, ...


# --------------------------------------------------------------------------
# OLE2 / Compound File Binary
# --------------------------------------------------------------------------
class CompoundFile:
    """Read-only reader for Compound File Binary (OLE2) containers.

    Parameters
    ----------
    source : str, Path or bytes
        Path to the file, or its bytes (for a container nested in a stream).

    Raises
    ------
    ValueError
        If the source does not start with the Compound File signature.
    """

    SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

    def __init__(self, source: str | Path | bytes) -> None:
        self._bytes = raw = source if isinstance(source, bytes) else Path(source).read_bytes()
        if raw[:8] != self.SIGNATURE:
            raise ValueError("not a Compound File Binary container")
        sector_shift, mini_sector_shift = struct.unpack_from("<HH", raw, 0x1E)
        self.sector_size = 1 << sector_shift
        self.mini_sector_size = 1 << mini_sector_shift
        (fat_sector_count, directory_start, _, self.mini_stream_cutoff, mini_fat_start,
         _, difat_start, difat_sector_count) = struct.unpack_from("<IIIIIIII", raw, 0x2C)
        ids_per_sector = self.sector_size // 4
        difat = list(struct.unpack_from("<109I", raw, 0x4C))
        sector = difat_start
        while sector < _SPECIAL_SECTOR and difat_sector_count:
            values = struct.unpack(f"<{ids_per_sector}I", self._sector(sector))
            difat += values[:-1]
            sector = values[-1]
            difat_sector_count -= 1
        self._fat: list[int] = []
        for sector in difat[:fat_sector_count]:
            self._fat += struct.unpack(f"<{ids_per_sector}I", self._sector(sector))
        directory = self._chain(directory_start)
        self._entries: list[dict[str, Any]] = []
        for index in range(len(directory) // 128):
            entry = directory[index * 128:(index + 1) * 128]
            name_length = struct.unpack_from("<H", entry, 64)[0]
            left, right, child = struct.unpack_from("<III", entry, 68)
            start, size = struct.unpack_from("<IQ", entry, 116)
            if self.sector_size == 512:
                size &= 0xFFFFFFFF
            self._entries.append(dict(
                name=entry[:max(name_length - 2, 0)].decode("utf-16le", errors="replace"),
                type=entry[66], left=left, right=right, child=child, start=start, size=size))
        root = self._entries[0]
        self._mini_stream = self._chain(root["start"])[:root["size"]] if root["start"] < _SPECIAL_SECTOR else b""
        mini_fat = self._chain(mini_fat_start) if mini_fat_start < _SPECIAL_SECTOR else b""
        self._mini_fat = list(struct.unpack(f"<{len(mini_fat) // 4}I", mini_fat))
        self._paths: dict[str, dict[str, Any]] = {}
        self._walk(root["child"], "")

    def _sector(self, sector: int) -> bytes:
        offset = (sector + 1) * self.sector_size
        return self._bytes[offset:offset + self.sector_size]

    def _chain(self, sector: int) -> bytes:
        parts, steps = [], 0
        while sector < _SPECIAL_SECTOR and steps <= len(self._fat):
            parts.append(self._sector(sector))
            sector = self._fat[sector]
            steps += 1
        return b"".join(parts)

    def _mini_chain(self, sector: int) -> bytes:
        parts, steps = [], 0
        while sector < _SPECIAL_SECTOR and steps <= len(self._mini_fat):
            offset = sector * self.mini_sector_size
            parts.append(self._mini_stream[offset:offset + self.mini_sector_size])
            sector = self._mini_fat[sector]
            steps += 1
        return b"".join(parts)

    def _walk(self, index: int, prefix: str) -> None:
        stack = [(index, prefix)]
        while stack:
            index, prefix = stack.pop()
            if index >= _SPECIAL_SECTOR:
                continue
            entry = self._entries[index]
            stack += [(entry["left"], prefix), (entry["right"], prefix)]
            path = prefix + entry["name"]
            self._paths[path] = entry
            if entry["type"] == 1:
                stack.append((entry["child"], path + "/"))

    def list_streams(self) -> dict[str, int]:
        """Return every stream path with its size in bytes, sorted by path."""
        return {path: entry["size"] for path, entry in sorted(self._paths.items()) if entry["type"] == 2}

    def has_stream(self, path: str) -> bool:
        """Return whether a stream exists at ``path``.

        Parameters
        ----------
        path : str
            Stream path, storages separated by ``/``.

        Returns
        -------
        bool
            True if a stream (not a storage) exists there.
        """
        return path in self._paths and self._paths[path]["type"] == 2

    def read_stream(self, path: str) -> bytes:
        """Return the bytes of one stream.

        Parameters
        ----------
        path : str
            Stream path, storages separated by ``/``.

        Returns
        -------
        bytes
            The stream content.

        Raises
        ------
        KeyError
            If no entry exists at ``path``.
        """
        entry = self._paths[path]
        if entry["size"] < self.mini_stream_cutoff:
            raw = self._mini_chain(entry["start"])
        else:
            raw = self._chain(entry["start"])
        return raw[:entry["size"]]


# --------------------------------------------------------------------------
# stream decoders
# --------------------------------------------------------------------------
def decode_mfc_strings(raw: bytes) -> list[str]:
    """Return every MFC ``CString`` in a stream, in order.

    A ``CString`` is serialised as the marker ``ff fe ff``, a length prefix
    (``uint8``, escalating to ``uint16`` and ``uint32`` on ``0xff``) and that
    many UTF-16LE characters.

    Parameters
    ----------
    raw : bytes
        Stream content.

    Returns
    -------
    list[str]
        The decoded strings, empty ones included.
    """
    strings, position = [], 0
    while True:
        position = raw.find(b"\xff\xfe\xff", position)
        if position < 0 or position + 4 > len(raw):
            return strings
        length, start = raw[position + 3], position + 4
        if length == 0xFF:
            length, start = struct.unpack_from("<H", raw, start)[0], start + 2
            if length == 0xFFFF:
                length, start = struct.unpack_from("<I", raw, start)[0], start + 4
        strings.append(raw[start:start + 2 * length].decode("utf-16le", errors="replace"))
        position = start + 2 * length


def _filetime_to_iso(value: int) -> str | None:
    if value == 0:
        return None
    return (datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=value // 10)).isoformat()


_SUMMARY_PROPERTY_NAMES = {
    2: "title", 3: "subject", 4: "author", 5: "keywords", 6: "comments", 7: "template",
    8: "last_author", 9: "revision", 10: "edit_time", 11: "last_printed", 12: "created",
    13: "last_saved", 14: "pages", 15: "words", 16: "characters", 18: "application", 19: "security",
}


def decode_summary_information(raw: bytes) -> dict[str, Any]:
    """Decode an OLE ``SummaryInformation`` property set.

    Parameters
    ----------
    raw : bytes
        Content of a ``\\x05SummaryInformation`` stream.

    Returns
    -------
    dict
        Property name to value; timestamps as ISO 8601 strings in UTC.
    """
    if len(raw) < 48:
        return {}
    section_offset = struct.unpack_from("<I", raw, 44)[0]
    _, count = struct.unpack_from("<II", raw, section_offset)
    codepage = "cp1252"
    properties: dict[str, Any] = {}
    for index in range(count):
        property_id, offset = struct.unpack_from("<II", raw, section_offset + 8 + 8 * index)
        position = section_offset + offset
        variant_type = struct.unpack_from("<I", raw, position)[0]
        if variant_type == 2:
            value: Any = struct.unpack_from("<h", raw, position + 4)[0]
        elif variant_type == 3:
            value = struct.unpack_from("<i", raw, position + 4)[0]
        elif variant_type == 30:
            length = struct.unpack_from("<I", raw, position + 4)[0]
            value = raw[position + 8:position + 8 + length].split(b"\0")[0].decode(codepage, errors="replace")
        elif variant_type == 31:
            length = struct.unpack_from("<I", raw, position + 4)[0]
            value = raw[position + 8:position + 8 + 2 * length].decode("utf-16le", errors="replace").rstrip("\0")
        elif variant_type == 64:
            ticks = struct.unpack_from("<Q", raw, position + 4)[0]
            value = ticks / 1e7 if property_id == 10 else _filetime_to_iso(ticks)
        else:
            value = f"<variant type {variant_type}>"
        if property_id == 1 and variant_type == 2:
            codepage = f"cp{value & 0xFFFF}"
            continue
        properties[_SUMMARY_PROPERTY_NAMES.get(property_id, f"property_{property_id}")] = value
    return properties


def decode_trace(raw: bytes) -> dict[str, Any]:
    """Decode one ``Detector Data/Detector <id> Trace`` stream.

    Parameters
    ----------
    raw : bytes
        Stream content.

    Returns
    -------
    dict
        ``counts`` (raw ADC counts, float64 holding whole numbers, first one at
        t = 0), ``sampling_interval`` in s, ``scale_factor`` (detector units
        per count), and the undecoded ``header`` words and ``trailer`` bytes
        (hex) for provenance.

    Raises
    ------
    ValueError
        If the header does not have the known layout.
    """
    header = struct.unpack_from("<5i", raw, 0)
    point_count = header[2]
    if point_count != header[1] + 1:
        raise ValueError(f"unexpected trace header {header}")
    counts = np.frombuffer(raw, "<f8", count=point_count, offset=20).copy()
    trailer = raw[20 + 8 * point_count:]
    return dict(
        counts=counts,
        sampling_interval=round(float(struct.unpack_from("<f", trailer, 16)[0]), 7),
        scale_factor=struct.unpack_from("<f", trailer, 20)[0],
        header=list(header),
        trailer=trailer.hex(),
    )


def decode_audit_trail(raw: bytes) -> list[dict[str, Any]]:
    """Read the SQLite audit trail embedded in a ``.dat``, string ids resolved.

    Parameters
    ----------
    raw : bytes
        Content of the ``AuditTrail/AuditTrailDB`` stream (a SQLite database).

    Returns
    -------
    list[dict]
        One entry per log row, in order. Column names are the database's own.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "audit_trail.sqlite"
        path.write_bytes(raw)
        connection = sqlite3.connect(path)
        try:
            strings = dict(connection.execute("select ID, StringValue from StringTable"))
            rows = connection.execute(
                "select ID, ActivityDate, Location, UserInfo, Description, Reason from LogTable order by ID"
            ).fetchall()
        finally:
            connection.close()
    return [
        dict(id=row[0], date_utc=row[1], location=strings.get(row[2], row[2]), user=strings.get(row[3], row[3]),
             description=strings.get(row[4], row[4]), reason=strings.get(row[5], row[5]))
        for row in rows
    ]


# --------------------------------------------------------------------------
# .dat
# --------------------------------------------------------------------------
def read_dat(file_path: str | Path, include_method_xml: bool = True) -> dict[str, Any]:
    """Read one EZChrom injection file (``.dat``).

    Parameters
    ----------
    file_path : str or Path
        The ``.dat`` file inside an ``.rslt`` folder.
    include_method_xml : bool, optional
        Keep the instrument method XML in the metadata (default is True). It is
        about 50 000 characters and identical for every injection of a sequence.

    Returns
    -------
    dict
        ``"data"``: one entry per detector signal, keyed by signal name
        (``"Front Signal"``, ``"Back Signal"``), each with ``time`` in s,
        ``counts`` (raw ADC counts), ``signal`` (counts * scale factor, in
        ``unit``), ``unit``, ``scale_factor``, ``sampling_interval`` in s,
        ``detector_id`` and the undecoded trace header and trailer.
        ``"meta"``: header strings, OLE summary information of the file and of
        the embedded method, the Agilent 7890 instrument method XML,
        instrument setup text, the audit trail and the stream inventory.
    """
    file_path = Path(file_path)
    container = CompoundFile(file_path)
    streams = container.list_streams()
    meta: dict[str, Any] = {"file": str(file_path), "streams": streams}

    handler = decode_mfc_strings(container.read_stream("Detector Trace Handler"))
    detectors: dict[str, dict[str, str]] = {}
    for start in range(0, len(handler) - 6, 7):
        detectors[handler[start]] = dict(name=handler[start + 1], unit=handler[start + 2], label=handler[start + 4])
    meta["detectors"] = detectors

    data: dict[str, Any] = {}
    for stream in streams:
        if not (stream.startswith("Detector Data/Detector ") and stream.endswith(" Trace")):
            continue
        detector_id = stream.split()[-2]
        trace = decode_trace(container.read_stream(stream))
        detector = detectors.get(detector_id, dict(name=f"Detector {detector_id}", unit=""))
        data[detector["name"]] = dict(
            detector_id=detector_id,
            time=np.arange(len(trace["counts"])) * trace["sampling_interval"],
            counts=trace["counts"],
            signal=trace["counts"] * trace["scale_factor"],
            unit=detector["unit"],
            scale_factor=trace["scale_factor"],
            sampling_interval=trace["sampling_interval"],
            trace_header=trace["header"],
            trace_trailer=trace["trailer"],
        )

    header_strings = decode_mfc_strings(container.read_stream("Chrom Header")) if container.has_stream("Chrom Header") else []
    meta["header"] = dict(zip(["method_path", "sample_name", "software_version", "user", "instrument_name"], header_strings))
    meta["header_strings"] = header_strings
    for key, stream in [("summary", "\x05SummaryInformation"),
                        ("method_summary", "Method10/Method/\x05SummaryInformation")]:
        if container.has_stream(stream):
            meta[key] = decode_summary_information(container.read_stream(stream))
    if container.has_stream("Contents"):
        meta["contents_strings"] = decode_mfc_strings(container.read_stream("Contents"))
    if container.has_stream("Method10/MethodInfo"):
        meta["method_info_strings"] = decode_mfc_strings(container.read_stream("Method10/MethodInfo"))
    if include_method_xml:
        for stream in streams:
            if stream.startswith("Method10/Method/DUIMethodSubStorage/") and not stream.endswith("AUX"):
                strings = decode_mfc_strings(container.read_stream(stream))
                xml = next((text for text in strings if text.lstrip().startswith("<?xml")), None)
                if xml:
                    meta.setdefault("instrument_method_xml", {})[stream.rsplit("/", 1)[-1]] = xml
    meta["instrument_setup"] = {
        stream.rsplit("/", 1)[-1]: [text for text in decode_mfc_strings(container.read_stream(stream)) if text.strip("-")]
        for stream in streams if stream.startswith("AnalysisServerInstrumentSetup/")
    }
    if container.has_stream("AuditTrail/AuditTrailDB"):
        meta["audit_trail"] = decode_audit_trail(container.read_stream("AuditTrail/AuditTrailDB"))
    if container.has_stream("Configuration"):
        meta["configuration_streams"] = CompoundFile(container.read_stream("Configuration")).list_streams()
    return {"data": data, "meta": meta}


def read_container_strings(file_path: str | Path) -> dict[str, list[str]]:
    """Return the ``CString`` text of every stream in an OLE2 container (``.rst``, ``.cfg``, ``.dat``).

    Parameters
    ----------
    file_path : str or Path
        Any Compound File Binary container.

    Returns
    -------
    dict[str, list[str]]
        Stream path to its strings, for streams holding at least one.
    """
    container = CompoundFile(file_path)
    strings = {stream: decode_mfc_strings(container.read_stream(stream)) for stream in container.list_streams()}
    return {stream: values for stream, values in strings.items() if values}


def dump_streams(file_path: str | Path, output_directory: str | Path) -> list[Path]:
    """Write every stream of an OLE2 container to its own file, for archiving or inspection.

    Parameters
    ----------
    file_path : str or Path
        Any Compound File Binary container.
    output_directory : str or Path
        Directory to write into; storages become subdirectories.

    Returns
    -------
    list[Path]
        The files written.
    """
    container = CompoundFile(file_path)
    output_directory = Path(output_directory)
    written = []
    for stream in container.list_streams():
        target = output_directory / (stream.replace("\x05", "_") + ".bin")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(container.read_stream(stream))
        written.append(target)
    return written


# --------------------------------------------------------------------------
# ACAML
# --------------------------------------------------------------------------
def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _flatten(element: ET.Element | None) -> dict[str, Any]:
    """Flatten an ACAML element one level: ``val`` attributes become numbers, leaf text stays text."""
    flat: dict[str, Any] = {}
    if element is None:
        return flat
    for child in element:
        key = _local_name(child.tag)
        if "val" in child.attrib:
            try:
                value: Any = float(child.attrib["val"])
            except ValueError:
                value = child.attrib["val"]
            if "unit" in child.attrib:
                flat[key + "_unit"] = child.attrib["unit"]
        elif len(child) == 0:
            value = (child.text or "").strip()
        else:
            continue
        if key in flat:
            flat[key] = flat[key] if isinstance(flat[key], list) else [flat[key]]
            flat[key].append(value)
        else:
            flat[key] = value
    return flat


def _custom_fields(element: ET.Element) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for field in element.findall("a:CustomField", ACAML_NAMESPACE):
        value = field.find("a:Value", ACAML_NAMESPACE)
        fields[field.get("Name")] = (value.text or "").strip() if value is not None else _custom_fields(field)
    return fields


def _reference_id(element: ET.Element, path: str) -> str | None:
    reference = element.find(path, ACAML_NAMESPACE)
    return reference.get("id") if reference is not None else None


def read_acaml(file_path: str | Path) -> dict[str, Any]:
    """Read an ACAML result file (``.acaml`` for a sequence, ``.acam_`` for one injection).

    Parameters
    ----------
    file_path : str or Path
        The ACAML file.

    Returns
    -------
    dict
        ``"data"``: ``injections``, one per injection with the ``.dat`` it
        belongs to, its signals, ``peaks`` (every integrator field; times in
        min, height and area in counts and counts s), ``species`` (amounts,
        response factors, the peak and calibration curve each was quantified
        from; the vendor calls them compounds) and ``diagnostics`` (column and liner); plus
        ``calibration_curves`` keyed by curve id, each with coefficients and
        levels with their calibration history.
        ``"meta"``: instrument, methods, samples, species names, document info.
    """
    namespace = ACAML_NAMESPACE
    root = ET.parse(file_path).getroot()
    content = root.find("a:Doc/a:Content", namespace)

    species_names = {entry.get("id"): entry.findtext("a:Name", "", namespace)
                     for entry in content.findall("a:Compound", namespace)}

    calibration_curves = {}
    for curve in content.findall("a:Resources/a:CalibrationCurve", namespace):
        record = _flatten(curve)
        record["coefficients"] = _flatten(curve.find("a:Coefficients", namespace))
        record["levels"] = []
        for level in curve.findall("a:CalibrationLevel", namespace):
            level_record = _flatten(level)
            level_record["history"] = [
                dict(_flatten(entry), user=entry.findtext("a:ProcessedBy/a:Username", "", namespace))
                for entry in level.findall("a:CalibrationHistory", namespace)
            ]
            record["levels"].append(level_record)
        calibration_curves[curve.get("id")] = record

    measurements = {measurement.get("id"): measurement
                    for measurement in content.findall("a:Injections/a:MeasData", namespace)}
    injections = []
    for result in content.findall("a:Injections/a:Result", namespace):
        measurement = measurements.get(_reference_id(result, "a:InjectionMeasData_ID"))
        signals: dict[str, dict[str, Any]] = {}
        dat_file = None
        if measurement is not None:
            dat_file = measurement.findtext("a:BinaryData/a:DataItem/a:Name", None, namespace)
            for signal in measurement.findall("a:Signal", namespace):
                record = _flatten(signal)
                record.pop("BinaryData", None)
                signals[signal.get("id")] = record
        peaks, signal_of_peak = [], {}
        for signal_result in result.findall("a:SignalResult", namespace):
            signal_name = signals.get(_reference_id(signal_result, "a:Signal_ID"), {}).get("Name")
            for peak in signal_result.findall("a:Peak", namespace):
                record = _flatten(peak)
                record["peak_id"] = peak.get("id")
                record["signal"] = signal_name
                record["BaselineParameters"] = [
                    float(parameter.get("val"))
                    for parameter in peak.findall("a:BaselineParameters/a:BaselineParameter", namespace)
                ]
                peaks.append(record)
                signal_of_peak[peak.get("id")] = signal_name
        species = []
        for injection_species in result.findall("a:InjectionCompound", namespace):
            record = _flatten(injection_species)
            record["species"] = species_names.get(_reference_id(injection_species, "a:IdentifiedCompound/a:Compound_ID"))
            peak_reference = injection_species.find("a:Identification/a:Qualified/a:Peaks/a:Peak_ID", namespace)
            record["peak_id"] = peak_reference.get("id") if peak_reference is not None else None
            record["signal"] = signal_of_peak.get(record["peak_id"])
            record["calibration_curve_id"] = (
                _reference_id(peak_reference, "a:CalibrationCurve_ID") if peak_reference is not None else None
            )
            species.append(record)
        species_of_peak = {record["peak_id"]: record["species"] for record in species if record["peak_id"]}
        for record in peaks:
            record["species"] = species_of_peak.get(record["peak_id"])
        injections.append(dict(
            dat_file=dat_file,
            created_utc=result.findtext("a:Info/a:CreatedDate", None, namespace),
            last_modified_utc=result.findtext("a:Info/a:LastModifiedDate", None, namespace),
            signals=list(signals.values()),
            peaks=peaks,
            species=species,
            diagnostics={item.get("Name"): _custom_fields(item)
                         for item in measurement.findall("a:DiagnosticData/a:Item", namespace)}
            if measurement is not None else {},
        ))

    instrument = content.find("a:Resources/a:Instrument", namespace)
    meta = dict(
        file=str(file_path),
        document=_flatten(root.find("a:Doc/a:DocInfo", namespace)),
        instrument=dict(name=instrument.findtext("a:Name", "", namespace),
                        **{f"module_{key}": value for key, value in _flatten(instrument.find("a:Module", namespace)).items()})
        if instrument is not None else {},
        methods=[_flatten(method) for method in content.findall("a:Method", namespace)],
        samples=[_flatten(sample) for sample in content.findall("a:SampleParams/a:IdentParam", namespace)],
        species=sorted(set(species_names.values())),
    )
    return {"data": {"injections": injections, "calibration_curves": calibration_curves}, "meta": meta}


def _method_section(section: ET.Element) -> dict[str, Any]:
    namespace = ACAML_NAMESPACE
    return dict(
        name=section.findtext("a:Name", "", namespace),
        id=section.findtext("a:ID", "", namespace),
        parameters=[
            dict(name=parameter.findtext("a:Name", "", namespace), id=parameter.findtext("a:ID", "", namespace),
                 value=parameter.findtext("a:Value", "", namespace), unit=parameter.findtext("a:Unit", "", namespace))
            for parameter in section.findall("a:Parameter", namespace)
        ],
        sections=[_method_section(child) for child in section.findall("a:Section", namespace)],
    )


def read_method_acaml(file_path: str | Path) -> dict[str, Any]:
    """Read an ACAML method configuration (``*_ACQ_*.macaml`` or ``*_DA_*.macaml``).

    Parameters
    ----------
    file_path : str or Path
        The ``.macaml`` file.

    Returns
    -------
    dict
        ``"data"``: the method as nested sections, each with ``name``, ``id``,
        ``parameters`` (``name``, ``id``, ``value``, ``unit``, all text, as
        written) and ``sections``. Lists, not dicts, because section names
        repeat (``Front Signal`` appears several times in one method).
        ``"meta"``: file and document info.
    """
    root = ET.parse(file_path).getroot()
    description = root.find("a:Doc/a:Content/a:MethodConfiguration/a:MethodDescription", ACAML_NAMESPACE)
    return {
        "data": _method_section(description) if description is not None else {},
        "meta": dict(file=str(file_path), document=_flatten(root.find("a:Doc/a:DocInfo", ACAML_NAMESPACE))),
    }


# --------------------------------------------------------------------------
# whole result set
# --------------------------------------------------------------------------
def read_result_set(directory: str | Path, include_method_xml: bool = False) -> dict[str, Any]:
    """Read a whole ``.rslt`` folder: every injection's raw traces and results, plus the sequence files.

    Parameters
    ----------
    directory : str or Path
        The ``... .rslt`` folder.
    include_method_xml : bool, optional
        Keep the instrument method XML on every injection (default is False).
        It is identical within a sequence, so by default it is kept once, in
        ``meta["instrument_method_xml"]``.

    Returns
    -------
    dict
        ``"data"``: ``injections``, a list in acquisition order, each a
        :func:`read_dat` dict extended by ``peaks``, ``species``,
        ``calibration_curves`` (only those used), ``result`` (result dates,
        ACAML signal records, diagnostics).
        ``"meta"``: the sequence ACAML metadata, both method ACAMLs, the
        strings of the ``.rst`` and ``.cfg`` containers, and the instrument
        method XML.
    """
    directory = Path(directory)
    injection_acamls = sorted(directory.glob("*.acam_"))
    sequence_acamls = sorted(directory.glob("*.acaml"))
    results: dict[str, dict[str, Any]] = {}
    curves: dict[str, dict[str, Any]] = {}
    for path in injection_acamls or sequence_acamls:
        parsed = read_acaml(path)
        curves.update(parsed["data"]["calibration_curves"])
        for injection in parsed["data"]["injections"]:
            if injection["dat_file"]:
                results[injection["dat_file"]] = injection

    injections = []
    method_xml = None
    for path in sorted(directory.glob("*.dat")):
        injection = read_dat(path, include_method_xml=True)
        if method_xml is None:
            method_xml = injection["meta"].get("instrument_method_xml")
        if not include_method_xml:
            injection["meta"].pop("instrument_method_xml", None)
        result = results.get(path.name, {})
        injection["peaks"] = result.get("peaks", [])
        injection["species"] = result.get("species", [])
        injection["result"] = {key: result.get(key) for key in ("created_utc", "last_modified_utc", "signals", "diagnostics")}
        used = {record["calibration_curve_id"] for record in injection["species"] if record.get("calibration_curve_id")}
        injection["calibration_curves"] = {key: value for key, value in curves.items() if key in used}
        injections.append(injection)

    meta: dict[str, Any] = {"directory": str(directory), "instrument_method_xml": method_xml}
    if sequence_acamls:
        meta["sequence"] = read_acaml(sequence_acamls[0])["meta"]
    meta["methods"] = {path.name: read_method_acaml(path)["data"] for path in sorted(directory.glob("*.macaml"))}
    meta["containers"] = {path.name: read_container_strings(path)
                          for path in sorted(list(directory.glob("*.rst")) + list(directory.glob("*.cfg")))}
    return {"data": {"injections": injections}, "meta": meta}


# --------------------------------------------------------------------------
# conversion to the common injection records of phd_parser.chromatography
# --------------------------------------------------------------------------
_MINUTES_TO_SECONDS = 60.0


def _number(text: str) -> float | None:
    """Parse a method parameter value; the method stores numbers with the instrument PC's decimal comma."""
    try:
        return float(str(text).strip().replace(",", "."))
    except ValueError:
        return None


def _walk_sections(section: dict[str, Any]):
    yield section
    for child in section.get("sections", []):
        yield from _walk_sections(child)


def detector_names(result_set: dict[str, Any]) -> dict[str, str]:
    """Map each vendor signal name to a detector name, from the acquisition method.

    The acquisition method describes each signal as e.g. ``"Front Signal (FID)"``,
    giving ``{"Front Signal": "fid", "Back Signal": "tcd"}``. A detector type used
    by more than one signal gets the signal's position (``fid_front``), and a
    signal the method does not describe keeps a slug of its own name.

    Parameters
    ----------
    result_set : dict
        Output of :func:`read_result_set`.

    Returns
    -------
    dict[str, str]
        Vendor signal name to detector name.
    """
    types: dict[str, str] = {}
    for method in result_set["meta"].get("methods", {}).values():
        for section in _walk_sections(method):
            if not section["name"].startswith("Signal #"):
                continue
            parameters = {parameter["name"]: parameter["value"] for parameter in section["parameters"]}
            details = parameters.get("Details", "")
            if "(" in details and parameters.get("Description"):
                types[parameters["Description"]] = details.rsplit("(", 1)[1].rstrip(")").strip().lower()
    signals = list(dict.fromkeys(
        name for injection in result_set["data"]["injections"] for name in injection["data"]
    ))
    names = {}
    for signal in signals:
        detector_type = types.get(signal)
        slug = re.sub(r"[^0-9a-z]+", "_", signal.lower()).strip("_") or "signal"
        if detector_type is None:
            names[signal] = slug
        elif list(types.values()).count(detector_type) > 1:
            names[signal] = f"{detector_type}_{slug.split('_')[0]}"
        else:
            names[signal] = detector_type
    return names


def _expected_species(result_set: dict[str, Any]) -> dict[str, list[tuple[float | None, str]]]:
    """Species name -> [(expected retention time in min, vendor signal)], from the data analysis method's peak tables."""
    expected: dict[str, list[tuple[float | None, str]]] = {}
    for method in result_set["meta"].get("methods", {}).values():
        for section in _walk_sections(method):
            for child in section.get("sections", []):
                if child["name"] != "Peak Table":
                    continue
                for entry in child["sections"]:
                    parameters = {parameter["name"]: parameter["value"] for parameter in entry["parameters"]}
                    retention_time = _number(parameters.get("Expected Retention Time", ""))
                    expected.setdefault(entry["name"], []).append((retention_time, section["name"]))
    return expected


def to_injections(result_set: dict[str, Any]) -> dict[str, Any]:
    """Convert a :func:`read_result_set` result into the common injection records of ``GCData``.

    Units are converted to SI and detector units: times from min to s, peak
    heights and baseline levels from counts to detector units, areas from
    counts s to detector units s (each times the signal's scale factor).
    Amounts are taken as the vendor computed them.

    Parameters
    ----------
    result_set : dict
        Output of :func:`read_result_set`.

    Returns
    -------
    dict
        ``"injections"``: records as described in
        :meth:`phd_parser.chromatography.GCData.from_injections`.
        ``"attrs"``: dataset attributes (vendor, software, instrument,
        analysis source, methods, calibration curves, instrument method XML).
    """
    names = detector_names(result_set)
    expected = _expected_species(result_set)
    injections = []
    calibration: dict[str, Any] = {}
    for injection in result_set["data"]["injections"]:
        scale = {signal: trace["scale_factor"] for signal, trace in injection["data"].items()}
        injection_time = injection["result"].get("created_utc")
        if injection_time is None:
            injection_time = injection["meta"].get("summary", {}).get("created")
            logger.warning(
                f"{Path(injection['meta']['file']).name}: no ACAML result, using the file creation time "
                "as injection time (it precedes the injection by some seconds)."
            )
        detectors = {
            names[signal]: dict(
                retention_time=trace["time"], values=trace["signal"], unit=trace["unit"],
                vendor_name=signal, scale_factor=trace["scale_factor"],
            )
            for signal, trace in injection["data"].items()
        }
        peaks = []
        for peak in injection["peaks"]:
            factor = scale.get(peak["signal"], np.nan)
            peaks.append(dict(
                detector=names.get(peak["signal"], peak["signal"]),
                species=peak.get("species") or "",
                retention_time=peak["RetentionTime"] * _MINUTES_TO_SECONDS,
                start_time=peak["BeginTime"] * _MINUTES_TO_SECONDS,
                end_time=peak["EndTime"] * _MINUTES_TO_SECONDS,
                baseline_start_time=peak["BaselineStart"] * _MINUTES_TO_SECONDS,
                baseline_end_time=peak["BaselineEnd"] * _MINUTES_TO_SECONDS,
                baseline_start_value=peak["LevelStart"] * factor,
                baseline_end_value=peak["LevelEnd"] * factor,
                height=peak["Height"] * factor,
                area=peak["Area"] * factor,
                area_percent=peak.get("AreaPercent"),
            ))
        species = []
        for record in injection["species"]:
            if not record.get("species"):
                continue  # unidentified peak, already in the peak table
            signal = record.get("signal")
            if signal is None:  # expected but not found: the method's peak table says which signal
                candidates = expected.get(record["species"], [])
                if candidates:
                    target = record.get("ExpectedRetTime")
                    signal = min(candidates, key=lambda candidate: abs((candidate[0] or 0) - (target or 0)))[1]
            if signal is None:
                logger.warning(f"Species '{record['species']}' has no peak and is not in the method; skipped.")
                continue
            factor = scale.get(signal, np.nan)
            species.append(dict(
                detector=names.get(signal, signal),
                species=record["species"],
                amount=record.get("Amount"),
                amount_unit=record.get("Amount_unit", ""),
                area=None if record.get("Area") is None else record["Area"] * factor,
                height=None if record.get("Height") is None else record["Height"] * factor,
            ))
        calibration.update(injection.get("calibration_curves", {}))
        injections.append(dict(
            injection_time=injection_time,
            sample_name=injection["meta"]["header"].get("sample_name", ""),
            source_file=Path(injection["meta"]["file"]).name,
            detectors=detectors,
            peaks=peaks,
            species=species,
        ))

    meta = result_set["meta"]
    first = result_set["data"]["injections"][0]["meta"] if result_set["data"]["injections"] else {}
    attrs = dict(
        vendor="Agilent",
        software=f"OpenLab CDS EZChrom {first.get('header', {}).get('software_version', '')}".strip(),
        instrument=meta.get("sequence", {}).get("instrument", {}),
        analysis_source=f"OpenLab CDS EZChrom {first.get('header', {}).get('software_version', '')}".strip(),
        method_path=first.get("header", {}).get("method_path", ""),
        detector_names=names,
        methods=meta.get("methods", {}),
        calibration=calibration,
        instrument_method_xml=meta.get("instrument_method_xml") or {},
    )
    return {"injections": injections, "attrs": attrs}
