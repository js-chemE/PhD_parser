"""Vendor-independent container for gas chromatography data: chromatograms plus their analysis."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, field_validator

from phd_parser._selection import EmptyWindow, window_mask
from phd_parser._time import TimeOriginMixin, TimestampLike, _to_timestamp
from phd_parser.history import HistoryMixin
from phd_parser.history import (
    append_history,
    decode_attr as _decode_attr,
    encode_attr as _encode_attr,
    records_step,
    suppress_recording,
    to_json_safe,
)
from phd_parser.naming.species.convention import ConventionLike, Unknown, get_convention

logger = logging.getLogger(__name__)

_RETENTION_TIME_PREFIX = "retention_time_"

PEAK_COLUMNS: dict[str, str] = {
    # column name in the common peak record -> unit of the stored value
    "retention_time": "s",
    "start_time": "s",
    "end_time": "s",
    "baseline_start_time": "s",
    "baseline_end_time": "s",
    "baseline_start_value": "detector unit",
    "baseline_end_value": "detector unit",
    "height": "detector unit",
    "area": "detector unit s",
    "area_percent": "%",
}
SPECIES_QUANTITIES: tuple[str, ...] = ("amount", "area", "height")




def _retention_time_dim(detector: str) -> str:
    return _RETENTION_TIME_PREFIX + detector


def _axis_signature(retention_time: npt.NDArray) -> tuple[float, float]:
    """Sampling interval and start offset within one interval, the two numbers that decide stackability."""
    if retention_time.size < 2:
        return (float("nan"), float("nan"))
    interval = float(np.median(np.diff(retention_time)))
    offset = float(retention_time[0] / interval - np.round(retention_time[0] / interval))
    return (round(interval, 9), round(offset, 6))


class GCData(TimeOriginMixin, HistoryMixin, BaseModel):
    """Gas chromatography data: the chromatograms of a series of injections and their analysis.

    Immutable wrapper around an ``xr.Dataset``. Every processing method returns
    a new instance. A single measurement is an object with one injection.

    Layout
    ------
    dims:
        ``injection`` (always present, length 1 for a single measurement),
        ``retention_time_<detector>`` (one per detector, so detectors may be
        sampled differently), ``peak`` and ``species`` (only with an analysis).
    coords:
        ``injection`` (int), ``tos`` (s, along ``injection``),
        ``sample_name`` and ``source_file`` (along ``injection``),
        ``retention_time_<detector>`` (s).
    chromatograms:
        One variable per detector, named by detector (``fid``, ``tcd``), dims
        ``(injection, retention_time_<detector>)``, values in detector units.
        Attrs: ``unit``, ``vendor_name``, ``scale_factor`` (detector units per
        raw count, when known), ``sampling_interval`` (s). Injections shorter
        than the longest one are padded with NaN.
    analysis (optional):
        ``peak_*`` variables along ``peak``: ``peak_injection`` (value of the
        ``injection`` coord), ``peak_detector``, ``peak_species`` (empty for
        unidentified peaks) and the quantities of :data:`PEAK_COLUMNS`
        (times in s, heights in detector units, areas in detector units s).
        ``amount``, ``area`` and ``height`` with dims ``(injection, species)``,
        one entry per chemical species and detector it was quantified on (coords
        ``species_name`` and ``species_detector``).
    attrs:
        ``tos_start`` (ISO string), ``analysis_source``, ``vendor``,
        ``software``, ``instrument``, ``method``, ``calibration``,
        ``provenance`` and anything else the vendor converter forwards.
    """

    def _new(self, ds: xr.Dataset) -> "GCData":
        """A new instance around ``ds``."""
        return GCData(ds=ds)


    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    ds: xr.Dataset = Field(description="Dataset with an 'injection' dim and one chromatogram variable per detector.")

    @field_validator("ds", mode="before")
    @classmethod
    def _validate_dataset(cls, value: Any) -> xr.Dataset:
        if not isinstance(value, xr.Dataset):
            raise TypeError(f"'ds' must be an xr.Dataset, got {type(value)}")
        if "injection" not in value.dims:
            raise ValueError("Dataset must have an 'injection' dimension.")
        detectors = [name for name in value.data_vars if any(str(dim).startswith(_RETENTION_TIME_PREFIX) for dim in value[name].dims)]
        if not detectors:
            raise ValueError("Dataset must hold at least one chromatogram variable.")
        for name in detectors:
            expected = ("injection", _retention_time_dim(name))
            if value[name].dims != expected:
                raise ValueError(f"Chromatogram '{name}' must have dims {expected}, got {value[name].dims}")
        return value

    # ----------------------------------------------------------------
    # Core properties
    # ----------------------------------------------------------------

    @property
    def detectors(self) -> list[str]:
        """Names of the detectors, in dataset order."""
        return [name for name in self.ds.data_vars if self.ds[name].dims[-1] == _retention_time_dim(str(name))]

    @property
    def n_injections(self) -> int:
        """Number of injections."""
        return int(self.ds.sizes["injection"])

    def __len__(self) -> int:
        return self.n_injections

    @property
    def injection(self) -> npt.NDArray:
        """Values of the ``injection`` coordinate."""
        return self.ds.coords["injection"].values

    @property
    def sample_names(self) -> Optional[npt.NDArray]:
        """Sample name of every injection, or ``None`` if not recorded."""
        return self.ds.coords["sample_name"].values if "sample_name" in self.ds.coords else None

    @property
    def tos(self) -> Optional[npt.NDArray]:
        """Time on stream of every injection in seconds since ``tos_start``, or ``None``."""
        return self.ds.coords["tos"].values if "tos" in self.ds.coords else None

    @property
    def has_analysis(self) -> bool:
        """Whether peak and species results are stored."""
        return "peak_injection" in self.ds or "amount" in self.ds

    @property
    def analysis_source(self) -> Optional[str]:
        """Who produced the stored analysis, e.g. the vendor software, or ``None``."""
        return self.ds.attrs.get("analysis_source")

    def get_unit(self, detector: str) -> str:
        """Return the unit of a detector's signal.

        Parameters
        ----------
        detector : str
            Detector name, see :attr:`detectors`.

        Returns
        -------
        str
            The unit, empty if unknown.
        """
        return str(self._detector(detector).attrs.get("unit", ""))

    def get_retention_time(self, detector: str) -> npt.NDArray:
        """Return a detector's retention time axis in seconds.

        Parameters
        ----------
        detector : str
            Detector name, see :attr:`detectors`.

        Returns
        -------
        numpy.ndarray
            1-D retention times in s.
        """
        return self.ds.coords[_retention_time_dim(detector)].values

    def _detector(self, detector: str) -> xr.DataArray:
        if detector not in self.detectors:
            raise KeyError(f"No detector '{detector}'. Available: {self.detectors}")
        return self.ds[detector]

    def _position(self, injection: int) -> int:
        positions = np.flatnonzero(self.injection == injection)
        if positions.size == 0:
            raise KeyError(f"No injection {injection}. Available: {self.injection.min()}-{self.injection.max()}")
        return int(positions[0])

    # ----------------------------------------------------------------
    # Chromatograms
    # ----------------------------------------------------------------

    def get_chromatogram(self, detector: str, injection: Optional[int] = None) -> xr.DataArray:
        """Return chromatograms of one detector as a labelled array.

        Parameters
        ----------
        detector : str
            Detector name, see :attr:`detectors`.
        injection : int or None, optional
            Value of the ``injection`` coordinate. ``None`` (default) returns
            all injections as a 2-D array.

        Returns
        -------
        xarray.DataArray
            Dims ``(injection, retention_time_<detector>)``, or the 1-D
            retention time dim for one injection. Slices like a numpy array
            (``chromatogram[3, 100:200]``, ``chromatogram.values``) and keeps
            ``tos`` and the retention times as coordinates.
        """
        data = self._detector(detector)
        if injection is None:
            return data
        return data.isel(injection=self._position(injection))

    # ----------------------------------------------------------------
    # Analysis
    # ----------------------------------------------------------------

    def get_peaks(self) -> pd.DataFrame:
        """Return the peak table, one row per integrated peak.

        Returns
        -------
        pandas.DataFrame
            Columns ``injection``, ``tos`` (s, if known), ``detector``,
            ``species`` and the quantities of :data:`PEAK_COLUMNS` in SI
            units. Empty if no analysis is stored.
        """
        if "peak_injection" not in self.ds:
            return pd.DataFrame(columns=["injection", "detector", "species", *PEAK_COLUMNS])
        columns = ["injection", "detector", "species", *PEAK_COLUMNS]
        table = pd.DataFrame({column: self.ds[f"peak_{column}"].values for column in columns if f"peak_{column}" in self.ds})
        table["injection"] = table["injection"].astype(np.int64)
        if self.tos is not None:
            tos_of = dict(zip(self.injection.tolist(), self.tos.tolist()))
            table.insert(1, "tos", table["injection"].map(tos_of))
        return table

    def get_amounts(self, detector: Optional[str] = None, quantity: str = "amount") -> pd.DataFrame:
        """Return a species quantity for every injection as a wide table.

        Parameters
        ----------
        detector : str or None, optional
            Only this detector's species, with species names as columns.
            ``None`` (default) returns all, with ``(detector, species)`` columns.
        quantity : {"amount", "area", "height"}, optional
            Which stored quantity (default is "amount").

        Returns
        -------
        pandas.DataFrame
            Indexed by ``injection``, with ``tos`` in s as the first column if known.

        Raises
        ------
        ValueError
            If no species results are stored or the quantity is unknown.
        """
        if quantity not in SPECIES_QUANTITIES:
            raise ValueError(f"quantity must be one of {SPECIES_QUANTITIES}")
        if quantity not in self.ds:
            raise ValueError("No species results stored.")
        data = self.ds[quantity]
        detectors = self.ds.coords["species_detector"].values
        names = self.ds.coords["species_name"].values
        if detector is not None:
            self._detector(detector)
            keep = detectors == detector
            columns: Any = list(names[keep])
            values = data.values[:, keep]
        else:
            columns = pd.MultiIndex.from_arrays([detectors, names], names=["detector", "species"])
            values = data.values
        table = pd.DataFrame(values, index=pd.Index(self.injection, name="injection"), columns=columns)
        if self.tos is not None:
            table.insert(0, "tos", self.tos)
        return table

    def get_evolution(self, species: str, detector: Optional[str] = None, quantity: str = "amount") -> xr.DataArray:
        """Return one species' quantity over the injections.

        Parameters
        ----------
        species : str
            Species name as stored (vendor name, or the convention's after
            :meth:`rename_species`).
        detector : str or None, optional
            Detector the species was quantified on. Required only when the
            species is quantified on more than one.
        quantity : {"amount", "area", "height"}, optional
            Which stored quantity (default is "amount").

        Returns
        -------
        xarray.DataArray
            Along ``injection``, with ``tos`` as coordinate.

        Raises
        ------
        KeyError
            If the species is not found or is ambiguous without ``detector``.
        """
        names = self.ds.coords["species_name"].values
        matches = names == species
        if detector is not None:
            matches &= self.ds.coords["species_detector"].values == detector
        found = np.flatnonzero(matches)
        if found.size != 1:
            on = sorted(self.ds.coords["species_detector"].values[names == species])
            raise KeyError(f"'{species}' found on detectors {on}; pass detector= to choose." if on else f"No species '{species}'.")
        return self.ds[quantity].isel(species=int(found[0]))

    # ----------------------------------------------------------------
    # Immutable: naming
    # ----------------------------------------------------------------

    @records_step
    def rename_species(self, convention: ConventionLike = None, unknown: Unknown = "keep") -> "GCData":
        """Return a new ``GCData`` with the species renamed to a naming convention.

        Renames the species of the amount grid and of the peak table; the
        chromatograms are untouched.

        Parameters
        ----------
        convention : None, str, Path, dict or SpeciesConvention, optional
            ``None`` (default) for the standard convention of
            :mod:`phd_parser.naming.species`, a JSON file, an already read
            JSON object, or a convention.
        unknown : {"keep", "raise", "none"}, optional
            What to do with species the convention does not know (default:
            keep the name and warn).

        Returns
        -------
        GCData
            New instance; ``attrs["species_convention"]`` names the convention.

        Raises
        ------
        ValueError
            If two species on the same detector end up with one name.
        """
        resolved = get_convention(convention)
        ds = self.ds.copy()
        if "species_name" in ds.coords:
            renamed = [resolved.get_canonical(str(name), unknown) for name in ds.coords["species_name"].values]
            pairs = list(zip(ds.coords["species_detector"].values.tolist(), renamed))
            clashes = sorted({pair for pair in pairs if pairs.count(pair) > 1})
            if clashes:
                raise ValueError(f"Renaming merges species on the same detector: {clashes}")
            ds = ds.assign_coords(species_name=("species", np.array(renamed, dtype=object)))
        if "peak_species" in ds:
            ds["peak_species"] = ("peak", np.array(
                [resolved.get_canonical(str(name), unknown) if name else "" for name in ds["peak_species"].values],
                dtype=object,
            ))
        ds.attrs = {**self.ds.attrs, "species_convention": resolved.name}
        return GCData(ds=ds)

    # ----------------------------------------------------------------
    # Immutable: selection
    # ----------------------------------------------------------------

    def _select_positions(self, positions: npt.NDArray) -> "GCData":
        ds = self.ds.isel(injection=positions)
        if "peak_injection" in ds:
            ds = ds.isel(peak=np.isin(ds["peak_injection"].values, ds.coords["injection"].values))
        return GCData(ds=ds)

    @records_step
    def select_by_idx(self, index: Union[int, slice, Sequence[int]]) -> "GCData":
        """Return a new ``GCData`` with the injections at the given positions.

        Parameters
        ----------
        index : int, slice or sequence of int
            Positions along the injection axis, numpy style. An int keeps the
            injection dim with length 1.

        Returns
        -------
        GCData
            The selected injections, with their peaks.
        """
        positions = np.arange(self.n_injections)[index]
        return self._select_positions(np.atleast_1d(positions))

    @records_step
    def select_tos_range(
        self, min_s: Optional[float] = None, max_s: Optional[float] = None, empty: EmptyWindow = "warn"
    ) -> "GCData":
        """Return a new ``GCData`` restricted to a time-on-stream window.

        Parameters
        ----------
        min_s, max_s : float or None, optional
            Inclusive bounds in seconds; ``None`` means unbounded.
        empty : {"warn", "raise"}, optional
            What to do when nothing falls inside the window: log it and
            return the empty selection (default), or raise ``ValueError``.

        Returns
        -------
        GCData
            The injections inside the window, with their peaks.

        Raises
        ------
        ValueError
            If no ``tos`` is set, or if the window is empty and ``empty="raise"``.
        """
        if self.tos is None:
            raise ValueError("Cannot select by tos because no 'tos' coordinate is set.")
        mask = window_mask(self.tos, min_s, max_s, what="injection", axis="tos", unit="s", empty=empty)
        return self._select_positions(np.flatnonzero(mask))

    @records_step
    def select_retention_time_range(self, start_seconds: Optional[float] = None, end_seconds: Optional[float] = None,
                                    detector: Optional[str] = None) -> "GCData":
        """Return a new ``GCData`` with the chromatograms cut to a retention time window.

        The analysis is kept unchanged.

        Parameters
        ----------
        start_seconds, end_seconds : float or None, optional
            Inclusive bounds in s; ``None`` means unbounded.
        detector : str or None, optional
            Cut only this detector (default: all).

        Returns
        -------
        GCData
            New instance with the shortened axes.
        """
        selection = {}
        for name in [detector] if detector else self.detectors:
            axis = self.get_retention_time(name)
            keep = np.ones(axis.size, dtype=bool)
            if start_seconds is not None:
                keep &= axis >= start_seconds
            if end_seconds is not None:
                keep &= axis <= end_seconds
            selection[_retention_time_dim(name)] = np.flatnonzero(keep)
        return GCData(ds=self.ds.isel(selection))

    # ----------------------------------------------------------------
    # Immutable: combining
    # ----------------------------------------------------------------

    @records_step
    def merge(self, other: "GCData") -> "GCData":
        """Append the injections of another ``GCData`` along the injection axis.

        Both are converted to injection records and rebuilt, so the rules of
        :meth:`from_injections` apply: per detector the axes must share their
        sampling interval and grid, shorter injections are padded with NaN.
        ``other`` is rebased onto this object's ``tos_start``. Attributes of
        ``self`` win; ``other``'s are kept under ``provenance``.

        Parameters
        ----------
        other : GCData
            Injections to append.

        Returns
        -------
        GCData
            The combined series, injections renumbered from 0 in time order.

        Raises
        ------
        ValueError
            If a detector's axes cannot be stacked, or only one side has a
            ``tos_start``.
        """
        if (self.tos_start is None) != (other.tos_start is None):
            raise ValueError("Both or neither GCData must have a tos_start to merge.")
        records = self.to_injections() + other.to_injections()
        attrs = dict(self.ds.attrs)
        provenance = list(attrs.get("provenance", [])) if isinstance(attrs.get("provenance"), list) else []
        provenance.append({"merged": {key: value for key, value in other.ds.attrs.items() if key != "tos_start"}})
        attrs["provenance"] = provenance
        return GCData.from_injections(records, attrs=attrs, tos_start=self.tos_start)

    # ----------------------------------------------------------------
    # Common record format (the contract between vendor parsers and GCData)
    # ----------------------------------------------------------------

    def to_injections(self) -> list[dict[str, Any]]:
        """Return the injections as common records, the input format of :meth:`from_injections`.

        Returns
        -------
        list[dict]
            One record per injection; see :meth:`from_injections` for the keys.
        """
        peaks = self.get_peaks()
        records = []
        timestamps = self.timestamps
        for position, injection in enumerate(self.injection.tolist()):
            detectors = {}
            for name in self.detectors:
                values = self.ds[name].values[position]
                valid = ~np.isnan(values)
                attrs = self.ds[name].attrs
                detectors[name] = dict(
                    retention_time=self.get_retention_time(name)[valid],
                    values=values[valid],
                    unit=attrs.get("unit", ""),
                    vendor_name=attrs.get("vendor_name", name),
                    scale_factor=attrs.get("scale_factor"),
                )
            species_records = []
            if "amount" in self.ds:
                for index in range(self.ds.sizes["species"]):
                    species_records.append(dict(
                        detector=str(self.ds.coords["species_detector"].values[index]),
                        species=str(self.ds.coords["species_name"].values[index]),
                        amount_unit=self.ds["amount"].attrs.get("unit", ""),
                        **{quantity: float(self.ds[quantity].values[position, index])
                           for quantity in SPECIES_QUANTITIES if quantity in self.ds},
                    ))
            peak_rows = peaks[peaks["injection"] == injection].drop(columns=["injection", "tos"], errors="ignore")
            records.append(dict(
                injection_time=timestamps[position] if timestamps is not None else None,
                tos=None if timestamps is not None or self.tos is None else float(self.tos[position]),
                sample_name=str(self.sample_names[position]) if self.sample_names is not None else "",
                source_file=str(self.ds.coords["source_file"].values[position]) if "source_file" in self.ds.coords else "",
                detectors=detectors,
                peaks=peak_rows.to_dict("records"),
                species=species_records,
            ))
        return records

    @classmethod
    @records_step
    def from_injections(cls, records: Sequence[dict[str, Any]], attrs: Optional[dict[str, Any]] = None,
                        tos_start: Optional[TimestampLike] = None) -> "GCData":
        """Build a ``GCData`` from common injection records, the format every vendor converter produces.

        Parameters
        ----------
        records : sequence of dict
            One per injection, with keys:

            - ``injection_time``: absolute time of the injection (timestamp-like or None)
            - ``tos``: s, used only when ``injection_time`` is None (optional)
            - ``sample_name``, ``source_file``: str (optional)
            - ``detectors``: ``{name: {"retention_time": s, "values": detector
              units, "unit": str, "vendor_name": str, "scale_factor": float or
              None}}``
            - ``peaks``: list of dicts with ``detector``, ``species`` and the
              keys of :data:`PEAK_COLUMNS` (optional)
            - ``species``: list of dicts with ``detector``, ``species``,
              ``amount``, ``amount_unit``, ``area``, ``height`` (optional)
        attrs : dict, optional
            Dataset attributes (vendor, instrument, method, analysis_source, ...).
        tos_start : timestamp-like, optional
            Origin of ``tos``. Defaults to the first injection time.

        Returns
        -------
        GCData
            Injections sorted by time and numbered from 0.

        Raises
        ------
        ValueError
            If a detector's axes differ in sampling interval or grid between
            injections. :func:`group_injections` splits such records into
            stackable groups.
        """
        if not records:
            raise ValueError("No injection records given.")
        times = [pd.Timestamp(record["injection_time"]) if record.get("injection_time") is not None else None for record in records]
        if all(time is not None for time in times):
            order = sorted(range(len(records)), key=lambda index: times[index])
            origin = pd.Timestamp(tos_start) if tos_start is not None else times[order[0]]
            tos = np.array([(_to_timestamp(times[index], origin) - origin).total_seconds() for index in order])
        else:
            order = list(range(len(records)))
            origin = pd.Timestamp(tos_start) if tos_start is not None else None
            has_tos = all(records[index].get("tos") is not None for index in order)
            tos = np.array([float(records[index]["tos"]) for index in order]) if has_tos else None
        records = [records[index] for index in order]

        groups = group_injections(records)
        if len(groups) > 1:
            raise ValueError(
                "Detector axes differ in sampling interval or grid between injections, so they cannot share one "
                f"array. Injection groups (positions): {groups}. Use group_injections() and build one GCData per group."
            )

        injection = np.arange(len(records))
        coords: dict[str, Any] = {"injection": injection}
        if tos is not None:
            coords["tos"] = ("injection", tos)
        coords["sample_name"] = ("injection", np.array([str(record.get("sample_name", "")) for record in records], dtype=object))
        coords["source_file"] = ("injection", np.array([str(record.get("source_file", "")) for record in records], dtype=object))

        data_vars: dict[str, Any] = {}
        detector_names = list(dict.fromkeys(name for record in records for name in record["detectors"]))
        for name in detector_names:
            data_vars[name] = _stack_detector(name, records)

        dataset_attrs = dict(attrs or {})
        if origin is not None:
            dataset_attrs["tos_start"] = origin.isoformat()
        ds = xr.Dataset(data_vars, coords=coords, attrs=dataset_attrs)
        ds = _add_analysis(ds, records)
        return cls(ds=ds)

    # ----------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------

    @classmethod
    @records_step
    def from_arrays(cls, retention_time: npt.NDArray, values: npt.NDArray, detector: str = "signal", unit: str = "",
                    injection_time: Optional[TimestampLike] = None, sample_name: str = "") -> "GCData":
        """Build a single-injection ``GCData`` from one chromatogram.

        Parameters
        ----------
        retention_time : numpy.ndarray
            Retention times in s.
        values : numpy.ndarray
            Signal in detector units, same length.
        detector : str, optional
            Detector name (default "signal").
        unit : str, optional
            Signal unit.
        injection_time : timestamp-like, optional
            Absolute injection time; becomes ``tos_start``.
        sample_name : str, optional
            Sample name.

        Returns
        -------
        GCData
            One injection, one detector, no analysis.
        """
        record = dict(
            injection_time=injection_time, tos=0.0, sample_name=sample_name,
            detectors={detector: dict(retention_time=np.asarray(retention_time, float), values=np.asarray(values, float),
                                      unit=unit, vendor_name=detector, scale_factor=None)},
        )
        return cls.from_injections([record])

    @classmethod
    @records_step
    def from_ezchrom_rslt(cls, paths: Union[str, Path, Sequence[Union[str, Path]]],
                          tos_start: Optional[TimestampLike] = None) -> "GCData":
        """Read one or more Agilent OpenLab EZChrom ``.rslt`` folders into one series.

        Parameters
        ----------
        paths : str, Path or sequence of them
            ``.rslt`` folders, e.g. a reduction and a reaction sequence.
        tos_start : timestamp-like, optional
            Origin of ``tos``; defaults to the first injection.

        Returns
        -------
        GCData
            All injections in time order, with the vendor analysis.

        Raises
        ------
        ValueError
            If the folders' detector axes cannot be stacked; use
            :meth:`from_ezchrom_rslt_groups` for such data.
        """
        records, attrs = _read_ezchrom(paths)
        return cls.from_injections(records, attrs=attrs, tos_start=tos_start)

    @classmethod
    def from_ezchrom_rslt_groups(cls, paths: Union[str, Path, Sequence[Union[str, Path]]],
                                 tos_start: Optional[TimestampLike] = None) -> list["GCData"]:
        """Like :meth:`from_ezchrom_rslt`, but returns one ``GCData`` per group of stackable injections.

        Never fails on differing detector axes: injections whose axes differ
        (a method changed mid-sequence) end up in separate objects that share
        one ``tos_start``.

        Returns
        -------
        list[GCData]
            One per group, in time order of each group's first injection.
        """
        records, attrs = _read_ezchrom(paths)
        times = [pd.Timestamp(record["injection_time"]) for record in records if record.get("injection_time") is not None]
        origin = pd.Timestamp(tos_start) if tos_start is not None else (min(times) if times else None)
        groups = group_injections(records)
        if len(groups) > 1:
            logger.warning(f"Detector axes differ between injections; building {len(groups)} GCData objects.")
        with suppress_recording():
            objects = [cls.from_injections([records[index] for index in group], attrs=attrs, tos_start=origin) for group in groups]
        parameters = {"paths": to_json_safe(paths), "tos_start": to_json_safe(tos_start)}
        return [append_history(obj, "from_ezchrom_rslt_groups", {**parameters, "group": number})
                for number, obj in enumerate(objects)]

    # ----------------------------------------------------------------
    # Output
    # ----------------------------------------------------------------

    def to_netcdf(self, filepath: Union[str, Path]) -> None:
        """Save to a NetCDF file, reloadable with :meth:`from_netcdf`.

        Chromatograms that are still raw counts times their ``scale_factor``
        are stored as int32 counts with that scale factor (CF packing), which
        is lossless and halves the size.

        Parameters
        ----------
        filepath : str or Path
            Destination file.
        """
        ds = self.ds.copy()
        ds.attrs = {key: _encode_attr(value) for key, value in ds.attrs.items()}
        encoding = {}
        for name in self.detectors:
            ds[name].attrs = {key: _encode_attr(value) for key, value in ds[name].attrs.items()}
            scale = ds[name].attrs.get("scale_factor")
            if scale:
                # never leave "scale_factor" as a variable attribute: xarray's CF decoder
                # would multiply the values by it on reload
                ds[name].attrs = {key: value for key, value in ds[name].attrs.items() if key != "scale_factor"}
                counts = ds[name].values / scale
                finite = counts[np.isfinite(counts)]
                if finite.size and np.allclose(finite, np.round(finite), atol=1e-6) and np.abs(finite).max() < 2**31 - 1:
                    encoding[name] = {"dtype": "int32", "scale_factor": float(scale), "_FillValue": np.int32(-(2**31))}
                else:
                    ds[name].attrs["counts_scale_factor"] = float(scale)
        ds.to_netcdf(filepath, engine="h5netcdf", encoding=encoding)
        logger.debug(f"Saved NetCDF -> {filepath}")

    @classmethod
    @records_step
    def from_netcdf(cls, filepath: Union[str, Path]) -> "GCData":
        """Reload a ``GCData`` saved with :meth:`to_netcdf`.

        Parameters
        ----------
        filepath : str or Path
            The NetCDF file.

        Returns
        -------
        GCData
            The reloaded instance.
        """
        with xr.open_dataset(filepath) as handle:
            packed = {name: handle[name].encoding.get("scale_factor") for name in handle.data_vars}
            ds = handle.load()
        ds.attrs = {key: _decode_attr(value) for key, value in ds.attrs.items()}
        # NetCDF3 has no int64: restore the integer index coordinates
        ds = ds.assign_coords({name: ds.coords[name].astype(np.int64) for name in ("injection", "peak", "species") if name in ds.coords})
        if "peak_injection" in ds:
            ds["peak_injection"] = ds["peak_injection"].astype(np.int64)
        for name in ds.data_vars:
            ds[name].attrs = {key: _decode_attr(value) for key, value in ds[name].attrs.items()}
            if packed.get(name) is not None:
                ds[name].attrs["scale_factor"] = float(packed[name])
            elif "counts_scale_factor" in ds[name].attrs:
                ds[name].attrs["scale_factor"] = float(ds[name].attrs.pop("counts_scale_factor"))
        return cls(ds=ds)

    def to_dataframe(self, detector: str) -> pd.DataFrame:
        """Return one detector's chromatograms as a wide table.

        Parameters
        ----------
        detector : str
            Detector name.

        Returns
        -------
        pandas.DataFrame
            Indexed by retention time in s, one column per injection.
        """
        data = self._detector(detector)
        return pd.DataFrame(data.values.T, index=pd.Index(self.get_retention_time(detector), name="retention_time"),
                            columns=pd.Index(self.injection, name="injection"))

    def __repr__(self) -> str:
        tos = self.tos
        span = f"tos={tos.min():.0f}-{tos.max():.0f} s" if tos is not None and tos.size else "no tos"
        detectors = ", ".join(
            f"{name} ({self.get_unit(name)}, {self.get_retention_time(name).size} points)" for name in self.detectors
        )
        analysis = f", analysis={self.analysis_source or 'yes'}" if self.has_analysis else ""
        return f"GCData(n_injections={self.n_injections}, {span}, detectors=[{detectors}]{analysis})"

    def __str__(self) -> str:
        return self.__repr__()


# --------------------------------------------------------------------------
# module helpers
# --------------------------------------------------------------------------
def group_injections(records: Sequence[dict[str, Any]]) -> list[list[int]]:
    """Split injection records into groups whose detector axes can share one array.

    Two injections are stackable when every detector they have is sampled at
    the same interval on the same grid; run length may differ.

    Parameters
    ----------
    records : sequence of dict
        Injection records as for :meth:`GCData.from_injections`.

    Returns
    -------
    list[list[int]]
        Positions of the records in each group, in order of first appearance.
    """
    groups: dict[tuple, list[int]] = {}
    for position, record in enumerate(records):
        key = tuple(sorted(
            (name, _axis_signature(np.asarray(detector["retention_time"], float)))
            for name, detector in record["detectors"].items()
        ))
        groups.setdefault(key, []).append(position)
    return list(groups.values())


def _stack_detector(name: str, records: Sequence[dict[str, Any]]) -> xr.DataArray:
    """Put one detector's chromatograms on a common grid, padding shorter or missing ones with NaN."""
    present = [record["detectors"][name] for record in records if name in record["detectors"]]
    axes = [np.asarray(detector["retention_time"], float) for detector in present]
    interval = np.median(np.diff(axes[0])) if axes[0].size > 1 else 1.0
    starts = [int(np.round(axis[0] / interval)) for axis in axes]
    ends = [start + axis.size for start, axis in zip(starts, axes)]
    first, last = min(starts), max(ends)
    if len(set(ends)) > 1 or len(set(starts)) > 1 or len(present) < len(records):
        logger.warning(f"Detector '{name}': injections differ in run length; padding with NaN to {last - first} points.")
    grid = (np.arange(first, last) * interval).round(9)
    values = np.full((len(records), last - first), np.nan)
    for row, record in enumerate(records):
        detector = record["detectors"].get(name)
        if detector is None:
            continue
        start = int(np.round(detector["retention_time"][0] / interval)) - first
        values[row, start:start + len(detector["values"])] = detector["values"]
    reference = present[0]
    attrs = {"unit": reference.get("unit", ""), "vendor_name": reference.get("vendor_name", name),
             "sampling_interval": float(round(interval, 9))}
    if reference.get("scale_factor"):
        attrs["scale_factor"] = float(reference["scale_factor"])
    dim = _retention_time_dim(name)
    return xr.DataArray(values, dims=("injection", dim), coords={dim: grid}, attrs=attrs)


def _add_analysis(ds: xr.Dataset, records: Sequence[dict[str, Any]]) -> xr.Dataset:
    """Add the peak table and the species grid from the records, if any record has them."""
    peaks = [dict(peak, injection=position) for position, record in enumerate(records) for peak in record.get("peaks", [])]
    if peaks:
        table = pd.DataFrame(peaks)
        ds = ds.assign_coords(peak=np.arange(len(table)))
        ds["peak_injection"] = ("peak", table["injection"].to_numpy(int))
        for column in ("detector", "species"):
            ds[f"peak_{column}"] = ("peak", table.get(column, pd.Series([""] * len(table))).fillna("").astype(str).to_numpy(object))
        for column, unit in PEAK_COLUMNS.items():
            if column in table:
                ds[f"peak_{column}"] = ("peak", pd.to_numeric(table[column], errors="coerce").to_numpy(float), {"unit": unit})

    species_index: dict[tuple[str, str], int] = {}
    for record in records:
        for entry in record.get("species", []):
            species_index.setdefault((entry["detector"], entry["species"]), len(species_index))
    if species_index:
        grid = {quantity: np.full((len(records), len(species_index)), np.nan) for quantity in SPECIES_QUANTITIES}
        unit = ""
        for position, record in enumerate(records):
            for entry in record.get("species", []):
                column = species_index[(entry["detector"], entry["species"])]
                unit = unit or entry.get("amount_unit", "")
                for quantity in SPECIES_QUANTITIES:
                    if entry.get(quantity) is not None:
                        grid[quantity][position, column] = entry[quantity]
        keys = list(species_index)
        ds = ds.assign_coords(
            species=np.arange(len(keys)),
            species_detector=("species", np.array([key[0] for key in keys], dtype=object)),
            species_name=("species", np.array([key[1] for key in keys], dtype=object)),
        )
        ds["amount"] = (("injection", "species"), grid["amount"], {"unit": unit})
        ds["area"] = (("injection", "species"), grid["area"], {"unit": "detector unit s"})
        ds["height"] = (("injection", "species"), grid["height"], {"unit": "detector unit"})
    return ds


def _read_ezchrom(paths: Union[str, Path, Sequence[Union[str, Path]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from phd_parser.chromatography.agilent import ezchrom

    if isinstance(paths, (str, Path)):
        paths = [paths]
    records: list[dict[str, Any]] = []
    attrs: dict[str, Any] = {}
    for path in paths:
        converted = ezchrom.to_injections(ezchrom.read_result_set(path))
        records += converted["injections"]
        if not attrs:
            attrs = converted["attrs"]
        else:
            attrs.setdefault("provenance", []).append({"source": str(path), "attrs_differ": converted["attrs"] != attrs})
    attrs["source_folders"] = [str(path) for path in paths]
    return records, attrs


def slugify_detector(name: str) -> str:
    """Turn a vendor signal name into a variable name (``"Front Signal"`` -> ``"front_signal"``)."""
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_") or "signal"
