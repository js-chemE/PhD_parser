"""The infrared readers: ``IRData.from_*`` builds a :class:`~phd_parser.spectroscopy.SpectroscopyData`.

``IRData`` is not a data class. It is the place where infrared spectra come
from: every ``from_*`` reads a file or another library's object and returns
a plain ``SpectroscopyData`` with ``technique = 'infrared'``. Processing,
NetCDF and everything else is on that class; there is nothing an infrared
spectrum can do that a Raman spectrum cannot.
"""

import logging
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from phd_parser.history import records_step
from phd_parser.spectroscopy.core import _OMNIC_VLABEL_TO_DATA_TYPE, DataType, SpectroscopyData

logger = logging.getLogger(__name__)

TECHNIQUE = "infrared"


class IRData:
    """The infrared readers; every ``from_*`` returns a :class:`SpectroscopyData` with ``technique = 'infrared'``.

    This is a namespace, not a data class: it is never instantiated. The
    generic constructors (``from_netcdf``, ``from_xarray``,
    ``from_spectrochempy``) are on ``SpectroscopyData`` itself.
    """

    def __init__(self) -> None:
        """Not a data class: use the ``from_*`` readers, they return a ``SpectroscopyData``."""
        raise TypeError("IRData is a reader namespace; call IRData.from_omnic_spa(...) or build a SpectroscopyData.")

    @classmethod
    @records_step
    def from_arrays(
        cls,
        wavenumber_per_cm: npt.NDArray,
        values: npt.NDArray,
        *,
        tos: Optional[npt.NDArray] = None,
        tos_start: Optional[Union[pd.Timestamp, str]] = None,
        data_type: Optional[DataType] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SpectroscopyData:
        """Build an infrared ``SpectroscopyData`` from raw NumPy arrays.

        :meth:`SpectroscopyData.from_arrays` with ``technique='infrared'``,
        so ``data_type`` defaults to ``'single_beam'``.

        Parameters
        ----------
        wavenumber_per_cm : numpy.ndarray
            1-D wavenumber axis in cm⁻¹.
        values : numpy.ndarray
            ``(n_wavenumber,)`` for a single spectrum or ``(n_scan, n_wavenumber)``
            for a series.
        tos : numpy.ndarray or None, optional
            Elapsed seconds per scan of a series (default is ``None``).
        tos_start : pandas.Timestamp or str or None, optional
            Absolute start time of the measurement (default is ``None``).
        data_type : str or None, optional
            What the values are (default is ``'single_beam'``).
        metadata : dict or None, optional
            Raw metadata of the source.

        Returns
        -------
        SpectroscopyData
            New instance.
        """
        return SpectroscopyData.from_arrays(
            wavenumber_per_cm, values, tos=tos, tos_start=tos_start, data_type=data_type,
            technique=TECHNIQUE, metadata=metadata,
        )

    @classmethod
    @records_step
    def from_omnic_spa(
        cls,
        filepath: Union[str, Path],
        *,
        wavenumber_to_si_factor: float = 100.0,
        delta_time_seconds: Optional[float] = None,
        tos_start: Optional[Union[pd.Timestamp, str]] = None,
        strict_tos_start: bool = True,
        backend: Literal["auto", "spectrochempy", "omnic"] = "auto",
    ) -> SpectroscopyData:
        """Load one or more Thermo OMNIC ``.spa`` files.

        A single file and a directory of files produce the same structure: a
        2-D ``(scan, wavenumber)`` instance with a ``tos`` coordinate, holding
        one scan per file and sorted by spectrum index.  Reading one spectrum
        is just the one-element case, so ``tos_start`` applies either way.

        Every argument after ``filepath`` is keyword-only: passing a second
        path positionally is a mistake (join it onto ``filepath`` instead).

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to a single ``.spa`` file or a directory containing multiple
            ``.spa`` files.
        wavenumber_to_si_factor : float, optional
            Multiplicative factor to convert the file's native x-axis to m⁻¹
            (default is ``100.0``, converting cm⁻¹ to m⁻¹).
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
        SpectroscopyData
            New instance loaded from the file(s), ``technique = 'infrared'``.

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
        from phd_parser.spectroscopy.infrared import omnic
        from phd_parser.spectroscopy.infrared import spectrochempy as _scp_parser

        if delta_time_seconds is not None and tos_start is not None:
            raise ValueError("Specify either 'delta_time_seconds' or 'tos_start', not both.")
        if backend not in ("auto", "spectrochempy", "omnic"):
            raise ValueError(
                f"Unknown backend {backend!r}; expected 'auto', 'spectrochempy' or 'omnic'."
            )

        if backend == "spectrochempy":
            raw = _scp_parser.read_spa(filepath, delta_time_seconds=delta_time_seconds, tos_start=tos_start)
        elif backend == "omnic":
            raw = omnic.read_spa(filepath, delta_time_seconds=delta_time_seconds, tos_start=tos_start)
        else:  # "auto"
            try:
                raw = _scp_parser.read_spa(filepath, delta_time_seconds=delta_time_seconds, tos_start=tos_start)
                logger.debug("from_omnic_spa: using spectrochempy backend")
            except ImportError:
                logger.debug("spectrochempy not available; falling back to built-in omnic parser")
                raw = omnic.read_spa(filepath, delta_time_seconds=delta_time_seconds, tos_start=tos_start)

        wavenumber_si = np.asarray(raw["data"]["x"]) * wavenumber_to_si_factor
        values = np.asarray(raw["data"]["v"], dtype=float)
        raw_tos = raw["data"].get("tos")
        tos = np.asarray(raw_tos, dtype=float) if raw_tos is not None else None

        attrs: dict[str, Any] = {"technique": TECHNIQUE}
        parsed_tos_start = SpectroscopyData._parse_tos_start(tos_start, raw["meta"].get("tos_start"), strict_tos_start)
        if parsed_tos_start is not None:
            attrs["tos_start"] = parsed_tos_start.isoformat()

        vlabel = raw["meta"].get("vlabel", "")
        data_type = _OMNIC_VLABEL_TO_DATA_TYPE.get(vlabel)
        if data_type is None:
            raise ValueError(
                f"Omnic vlabel '{vlabel}' has no known data_type mapping. "
                f"Known mappings: {list(_OMNIC_VLABEL_TO_DATA_TYPE)}. "
                "Add it to _OMNIC_VLABEL_TO_DATA_TYPE in spectroscopy/core.py."
            )
        attrs["data_type"] = data_type

        ds = SpectroscopyData._build_ds(wavenumber_si, values, tos=tos, attrs=attrs)
        return SpectroscopyData(ds=ds)
