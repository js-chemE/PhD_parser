"""The Raman readers: ``RamanData.from_*`` builds a :class:`~phd_parser.spectroscopy.SpectroscopyData`.

``RamanData`` is not a data class. It is the place where Raman spectra come
from: every ``from_*`` reads a file and returns a plain ``SpectroscopyData``
with ``technique = 'raman'``, the Raman shift on the ``wavenumber`` axis and
the excitation wavelength in ``ds.attrs``. The Raman names (``shift``,
``select_shift_range``, the scattered-light axes) and the RamanSPy bridge
are on ``SpectroscopyData`` itself.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from phd_parser.history import records_step
from phd_parser.spectroscopy.core import SpectroscopyData

logger = logging.getLogger(__name__)

TECHNIQUE = "raman"


class RamanData:
    """The Raman readers; every ``from_*`` returns a :class:`SpectroscopyData` with ``technique = 'raman'``.

    This is a namespace, not a data class: it is never instantiated. The
    generic constructors (``from_netcdf``, ``from_xarray``,
    ``from_spectrochempy``, ``from_ramanspy``) are on ``SpectroscopyData``.
    """

    def __init__(self) -> None:
        """Not a data class: use the ``from_*`` readers, they return a ``SpectroscopyData``."""
        raise TypeError("RamanData is a reader namespace; call RamanData.from_renishaw_wdf(...) or build a SpectroscopyData.")

    @classmethod
    @records_step
    def from_arrays(
        cls,
        shift_per_cm: npt.NDArray,
        values: npt.NDArray,
        *,
        excitation_wavelength_nm: Optional[float] = None,
        data_type: Optional[str] = None,
        tos: Optional[npt.NDArray] = None,
        tos_start: Optional[Union[pd.Timestamp, str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SpectroscopyData:
        """Build a Raman ``SpectroscopyData`` from raw NumPy arrays.

        :meth:`SpectroscopyData.from_arrays` with ``technique='raman'``, so
        ``data_type`` defaults to ``'intensity'``.

        Parameters
        ----------
        shift_per_cm : numpy.ndarray
            1-D Raman shift axis in cm⁻¹ (stored as the ``wavenumber`` axis).
        values : numpy.ndarray
            ``(n_shift,)`` for a single spectrum or ``(n_scan, n_shift)`` for
            a series.
        excitation_wavelength_nm : float or None, optional
            Excitation laser wavelength in nm; gives the scattered-light axes.
        data_type : str or None, optional
            What the values are, ``'intensity'``, ``'counts'``,
            ``'counts_per_second'`` or ``'arbitrary'`` (default is ``'intensity'``).
        tos : numpy.ndarray or None, optional
            Elapsed seconds per scan of a series (default is ``None``).
        tos_start : pandas.Timestamp or str or None, optional
            Absolute start time of the measurement (default is ``None``).
        metadata : dict or None, optional
            Raw metadata of the source.

        Returns
        -------
        SpectroscopyData
            New instance.
        """
        return SpectroscopyData.from_arrays(
            shift_per_cm, values, tos=tos, tos_start=tos_start, data_type=data_type,
            technique=TECHNIQUE, excitation_wavelength_nm=excitation_wavelength_nm, metadata=metadata,
        )

    @classmethod
    @records_step
    def from_btc655n_export(
        cls,
        filepath: Union[str, Path],
        y_key: str = "Raw data #1",
        remove_empty: bool = True,
    ) -> SpectroscopyData:
        """Load a B&W Tek BTC655N spectrometer export file.

        The BTC655N exports wavelength in nm; this constructor converts those
        values to the Raman shift using the excitation wavelength found in
        the file metadata.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to the BTC655N text export file.
        y_key : str, optional
            Column to use as the values (default "Raw data #1"), one of
            :data:`phd_parser.spectroscopy.raman.btc655n.Y_KEYS`.
        remove_empty : bool, optional
            Drop rows with no wavelength data when ``True`` (default).

        Returns
        -------
        SpectroscopyData
            New single-spectrum instance in ``intensity``.

        Raises
        ------
        ValueError
            When the export metadata does not contain ``laser_wavelength``.
        """
        from phd_parser.spectroscopy.raman import btc655n

        raw = btc655n.read_export(filepath, remove_empty=remove_empty)

        excitation_nm = raw["meta"].get("laser_wavelength")
        if excitation_nm is None:
            raise ValueError(
                "BTC655N export is missing 'laser_wavelength' in metadata. "
                "Cannot compute Raman shift without excitation wavelength."
            )

        # BTC655N reports the wavelength of the scattered light in nm; with the
        # laser wavelength it is stored as the Raman shift
        return SpectroscopyData.from_arrays(
            np.asarray(raw["data"]["Wavelength"], dtype=float),
            np.asarray(raw["data"][y_key], dtype=float),
            axis_unit="nm",
            technique=TECHNIQUE,
            excitation_wavelength_nm=float(excitation_nm),
            data_type="intensity",
            metadata=raw["meta"],
        )

    @classmethod
    @records_step
    def from_renishaw_txt(
        cls,
        filepath: Union[str, Path],
        excitation_wavelength_nm: float,
    ) -> SpectroscopyData:
        """Load a Renishaw WiRe plain-text export (shift already in cm⁻¹).

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to the Renishaw ``.txt`` export file.
        excitation_wavelength_nm : float
            Excitation laser wavelength in nm (not in the file).

        Returns
        -------
        SpectroscopyData
            New single-spectrum instance in ``intensity``.
        """
        from phd_parser.spectroscopy.raman import renishaw

        raw = renishaw.read_export_txt(Path(filepath))
        return cls.from_arrays(
            np.asarray(raw["data"]["wavenumber"], dtype=float),
            np.asarray(raw["data"]["intensity"], dtype=float),
            excitation_wavelength_nm=excitation_wavelength_nm,
            data_type="intensity",
            metadata=raw["meta"],
        )

    @classmethod
    @records_step
    def from_renishaw_wdf(cls, filepath: Union[str, Path]) -> SpectroscopyData:
        """Load a Renishaw WiRe binary WDF file.

        The excitation wavelength, the shift axis and the acquisition time
        come from the WDF header. Single spectra, time series (2-D, with
        ``tos`` from the per-spectrum times) and maps (``(y, x, wavenumber)``)
        are supported. The header fields are kept in ``metadata``.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to the ``.wdf`` file.

        Returns
        -------
        SpectroscopyData
            New instance in ``counts``.

        Raises
        ------
        ValueError
            When the x unit is not ``RamanShift``, the data unit not
            ``Counts``, or the axis length does not match the data.
        """
        from phd_parser.spectroscopy.raman import renishaw

        filepath = Path(filepath)
        wdf = renishaw.read_export_wdf(filepath)

        if wdf.x_unit != renishaw.UnitType.RamanShift:
            raise ValueError(
                f"Expected x_unit RamanShift, got {wdf.x_unit}. "
                "Cannot interpret data as Raman shift."
            )
        if wdf.data_unit != renishaw.UnitType.Counts:
            raise ValueError(f"Expected data_unit Counts, got {wdf.data_unit}.")

        # WDF stores the shift axis directly in cm^-1; laser_cm1 is the excitation in cm^-1
        shift_per_cm = np.asarray(wdf.wavenumber, dtype=float)
        excitation_nm = 1e7 / float(wdf.laser_cm1)
        data = np.asarray(wdf.data, dtype=float)
        metadata = renishaw.wdf_metadata(wdf)
        metadata["folder"] = str(filepath.parent)
        metadata["filename"] = filepath.name
        tos_start = wdf.acq_time
        tos = np.asarray(wdf.times, dtype=float) if wdf.times is not None else None

        if data.ndim == 3:  # map, WDF order (n_y, n_x, n_spectral)
            n_y, n_x, n_spec = data.shape
            if n_spec != shift_per_cm.size:
                raise ValueError(
                    f"WDF spectral axis length ({shift_per_cm.size}) does not match data.shape[-1] ({n_spec})"
                )
            return SpectroscopyData._from_raman_parts(
                shift_per_cm * 100.0, data, excitation_nm, "counts",
                tos_start=tos_start, metadata=metadata,
                map_coords={"x": np.arange(n_x), "y": np.arange(n_y)},
            )

        if data.ndim == 2 and tos is not None and tos.size != data.shape[0]:
            logger.warning(
                f"WDF holds {tos.size} time stamps for {data.shape[0]} spectra; dropping the 'tos' coordinate."
            )
            tos = None
        return cls.from_arrays(
            shift_per_cm,
            data,
            excitation_wavelength_nm=excitation_nm,
            data_type="counts",
            tos=tos if data.ndim == 2 else None,
            tos_start=tos_start,
            metadata=metadata,
        )
