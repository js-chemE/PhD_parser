"""Infrared spectra: :class:`IRData`, a :class:`~phd_parser.spectroscopy.SpectroscopyData` with the OMNIC readers.

Everything that processes the spectra (background and ``data_type``
conversions, baseline, merging, ...) is in ``SpectroscopyData``; this class
adds the readers of infrared files and ``technique = 'infrared'``.
"""

import logging
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional, Union

import numpy as np
import pandas as pd

from phd_parser.history import records_step
from phd_parser.spectroscopy.core import (  # noqa: F401 — re-exported names of the former IR core
    BACKGROUND_REFERENCED_TYPES,
    DataType,
    SpectroscopyData,
    TimestampLike,
    XLabel,
    _OMNIC_VLABEL_TO_DATA_TYPE,
    _to_timedelta,
    _to_timestamp,
)
from phd_parser.spectroscopy.infrared import omnic

logger = logging.getLogger(__name__)

# The former name of the data_type literal.
IRDataType = DataType


class IRData(SpectroscopyData):
    """Infrared spectra: a :class:`SpectroscopyData` read from infrared instruments.

    Spectra are stored in SI units (wavenumber in m⁻¹). All processing methods
    return a new ``IRData`` instance rather than mutating ``self``.

    Attributes
    ----------
    ds : xr.Dataset
        Dataset containing a ``"data"`` variable with dims ``('wavenumber',)``
        for a single spectrum or ``('scan', 'wavenumber')`` for a time series.
        Wavenumber coordinate is in m⁻¹. An optional ``'tos'`` coordinate
        (elapsed seconds, attached to the ``scan`` dim), a ``'background'``
        variable (1-D, single_beam), and a ``'baseline'`` variable (same dims
        as ``'data'``) may also be present.  Dataset-level attrs
        hold ``technique`` (``'infrared'``), ``data_type``,
        ``wavenumber_unit``, ``tos_start`` (ISO string), ``history``, and
        provenance keys written by processing methods.
    """

    _technique: ClassVar[Optional[str]] = "infrared"
    _default_data_type: ClassVar[Optional[str]] = "single_beam"

    # ----------------------------------------------------------------
    # Readers of infrared files
    # ----------------------------------------------------------------

    @classmethod
    @records_step
    def from_omnic_spa(
        cls,
        filepath: Union[str, Path],
        *,
        wavenumber_2SI_factor: float = 100.0,
        delta_time_seconds: Optional[float] = None,
        tos_start: Optional[Union[pd.Timestamp, str]] = None,
        strict_tos_start: bool = True,
        backend: Literal["auto", "spectrochempy", "omnic"] = "auto",
    ) -> "IRData":
        """Load one or more Thermo OMNIC ``.spa`` files into an IRData.

        A single file and a directory of files produce the same structure: a
        2-D ``(scan, wavenumber)`` instance with a ``tos`` coordinate, holding
        one scan per file and sorted by spectrum index.  Reading one spectrum
        is just the one-element case, so ``tos_start`` applies either way.

        Every argument after ``filepath`` is keyword-only — passing a second
        path positionally is a mistake (join it onto ``filepath`` instead).

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to a single ``.spa`` file or a directory containing multiple
            ``.spa`` files.
        wavenumber_2SI_factor : float, optional
            Multiplicative factor to convert the file's native x-axis to m⁻¹
            (default is ``100.0``, converting cm⁻¹ → m⁻¹).
        delta_time_seconds : float or None, optional
            Fixed time step between consecutive spectra used to build the
            ``tos`` coordinate when embedded timestamps are unavailable.
            Mutually exclusive with ``tos_start`` (default is ``None``).
        tos_start : pandas.Timestamp or str or None, optional
            Absolute start time used to compute ``tos`` from file timestamps.
            Mutually exclusive with ``delta_time_seconds`` (default is
            ``None``).
        strict_tos_start : bool, optional
            Raise ``ValueError`` if the ``tos_start`` from file metadata
            cannot be parsed when ``True``; log a warning and skip it when
            ``False`` (default is ``True``).
        backend : {"auto", "spectrochempy", "omnic"}, optional
            Low-level file reader to use.  ``"auto"`` (default) tries
            spectrochempy first and falls back to the built-in omnic parser
            when spectrochempy is not installed.  ``"spectrochempy"`` forces
            the spectrochempy reader (raises ``ImportError`` if not available).
            ``"omnic"`` forces the built-in binary parser.

        Returns
        -------
        IRData
            New instance loaded from the file(s).

        Raises
        ------
        ValueError
            If both ``delta_time_seconds`` and ``tos_start`` are provided, if
            ``backend`` is not one of the three known values, if the OMNIC
            y-axis label is not in the known mapping, or if
            ``strict_tos_start`` is ``True`` and ``tos_start`` cannot be
            parsed.
        FileNotFoundError
            If ``filepath`` does not exist or resolves to no ``.spa`` file.
        ImportError
            If ``backend="spectrochempy"`` and spectrochempy is not installed.
        """
        if delta_time_seconds is not None and tos_start is not None:
            raise ValueError("Specify either 'delta_time_seconds' or 'tos_start', not both.")
        if backend not in ("auto", "spectrochempy", "omnic"):
            raise ValueError(
                f"Unknown backend {backend!r}; expected 'auto', 'spectrochempy' or 'omnic'."
            )

        from phd_parser.spectroscopy.infrared import spectrochempy as _scp_parser

        if backend == "spectrochempy":
            raw = _scp_parser.read_spa(
                filepath, delta_time_seconds=delta_time_seconds, tos_start=tos_start
            )
        elif backend == "omnic":
            raw = omnic.read_spa(filepath, delta_time_seconds=delta_time_seconds, tos_start=tos_start)
        else:  # "auto"
            try:
                raw = _scp_parser.read_spa(
                    filepath, delta_time_seconds=delta_time_seconds, tos_start=tos_start
                )
                logger.debug("from_omnic_spa: using spectrochempy backend")
            except ImportError:
                logger.debug("spectrochempy not available; falling back to built-in omnic parser")
                raw = omnic.read_spa(
                    filepath, delta_time_seconds=delta_time_seconds, tos_start=tos_start
                )

        wavenumber_si = np.asarray(raw["data"]["x"]) * wavenumber_2SI_factor
        values = np.asarray(raw["data"]["v"], dtype=float)
        tos = np.asarray(raw["data"].get("tos"), dtype=float) if "tos" in raw["data"] else None

        # Parse tos_start from argument or attributes, with optional strictness
        parsed_tos_start: Optional[pd.Timestamp] = None
        if tos_start is not None:
            parsed_tos_start = pd.Timestamp(tos_start)
        elif (raw_ts := raw["meta"].get("tos_start")) is not None:
            try:
                parsed_tos_start = pd.Timestamp(raw_ts)
            except Exception as exc:
                if strict_tos_start:
                    raise ValueError(f"Could not parse tos_start '{raw_ts}': {exc}") from exc
                logger.warning(f"Ignoring unparseable tos_start '{raw_ts}': {exc}")

        attrs = {}
        if parsed_tos_start is not None:
            attrs["tos_start"] = parsed_tos_start.isoformat()

        vlabel = raw["meta"].get("vlabel", "")
        data_type = _OMNIC_VLABEL_TO_DATA_TYPE.get(vlabel)
        if data_type is None:
            raise ValueError(
                f"Omnic vlabel '{vlabel}' has no known data_type mapping. "
                f"Known mappings: {list(_OMNIC_VLABEL_TO_DATA_TYPE)}. "
                "Add it to _OMNIC_VLABEL_TO_DATA_TYPE in infrared/core.py."
            )
        attrs["data_type"] = data_type

        ds = cls._build_ds(
            wavenumber_si,
            values,
            tos=tos,
            attrs=attrs,
        )

        return cls(ds=ds)

    @classmethod
    @records_step
    def from_scp(
        cls,
        nd,
        *,
        wavenumber_2SI_factor: float = 100.0,
        delta_time_seconds: Optional[float] = None,
        tos_start: Optional[Union[pd.Timestamp, str]] = None,
        strict_tos_start: bool = True,
    ) -> "IRData":
        """Construct an ``IRData`` from a spectrochempy ``NDDataset``.

        Accepts any ``NDDataset`` — freshly read from a file, the result of
        spectrochempy processing, or built programmatically — and converts it
        to an immutable ``IRData``.  The wavenumber axis must be in cm⁻¹
        (the default for all OMNIC and most other spectrochempy readers).

        Parameters
        ----------
        nd : spectrochempy.NDDataset
            Source dataset.  ``nd.x`` is taken as the wavenumber axis and
            ``nd.data`` as the spectral values.
        wavenumber_2SI_factor : float, optional
            Multiplicative factor converting the dataset's x-axis to m⁻¹
            (default is ``100.0``, for cm⁻¹ → m⁻¹).
        delta_time_seconds : float or None, optional
            Fixed time step between consecutive scans used to build the
            ``tos`` coordinate.  Mutually exclusive with ``tos_start``
            (default is ``None``).
        tos_start : pandas.Timestamp or str or None, optional
            Absolute start time used to anchor tos values extracted from
            the dataset's y-axis timestamps.  Mutually exclusive with
            ``delta_time_seconds`` (default is ``None``).
        strict_tos_start : bool, optional
            Raise ``ValueError`` when ``tos_start`` cannot be parsed if
            ``True``; log a warning and continue if ``False`` (default is
            ``True``).

        Returns
        -------
        IRData
            New 2-D ``(scan, wavenumber)`` instance, also for a single-scan
            dataset — one scan is just the one-element case of a series.

        Raises
        ------
        ValueError
            If both ``delta_time_seconds`` and ``tos_start`` are supplied, or
            if the dataset title has no known ``data_type`` mapping.
        """
        from phd_parser.spectroscopy.infrared import spectrochempy as _scp_parser

        raw = _scp_parser.read_nddataset(
            nd, delta_time_seconds=delta_time_seconds, tos_start=tos_start
        )

        wavenumber_si = np.asarray(raw["data"]["x"]) * wavenumber_2SI_factor
        values = np.asarray(raw["data"]["v"], dtype=float)
        tos = np.asarray(raw["data"]["tos"], dtype=float) if "tos" in raw["data"] else None

        parsed_tos_start: Optional[pd.Timestamp] = None
        if tos_start is not None:
            parsed_tos_start = pd.Timestamp(tos_start)
        elif (raw_ts := raw["meta"].get("tos_start")) is not None:
            try:
                parsed_tos_start = pd.Timestamp(raw_ts)
            except Exception as exc:
                if strict_tos_start:
                    raise ValueError(f"Could not parse tos_start '{raw_ts}': {exc}") from exc
                logger.warning("Ignoring unparseable tos_start '%s': %s", raw_ts, exc)

        attrs: dict[str, Any] = {}
        if parsed_tos_start is not None:
            attrs["tos_start"] = parsed_tos_start.isoformat()

        vlabel = raw["meta"].get("vlabel", "")
        data_type = _OMNIC_VLABEL_TO_DATA_TYPE.get(vlabel)
        if data_type is None:
            raise ValueError(
                f"spectrochempy dataset title '{vlabel}' has no known data_type mapping. "
                f"Known mappings: {list(_OMNIC_VLABEL_TO_DATA_TYPE)}."
            )
        attrs["data_type"] = data_type

        ds = cls._build_ds(wavenumber_si, values, tos=tos, attrs=attrs)
        return cls(ds=ds)
