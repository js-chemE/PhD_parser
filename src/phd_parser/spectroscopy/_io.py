"""NetCDF, CSV and the bridges to SpectroChemPy and RamanSPy.

A mixin of :class:`~phd_parser.spectroscopy.core.SpectroscopyData`; never used on its own.
"""

import logging
from typing import TYPE_CHECKING
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from phd_parser.history import HISTORY_KEY, decode_attrs, encode_attrs, records_step, to_json_safe
from phd_parser.spectroscopy._common import (
    _DATA_TYPE_TO_SCP_TITLE,
    _OMNIC_VLABEL_TO_DATA_TYPE,
    DataType,
    _data_type_from_title,
    _dataset_from_dataarray,
    _technique_of_data_type,
)

if TYPE_CHECKING:
    from phd_parser.spectroscopy.core import SpectroscopyData

logger = logging.getLogger(__name__)


class _IOMixin:
    """NetCDF, CSV and the bridges to SpectroChemPy and RamanSPy."""

    def _netcdf_attrs(self) -> dict[str, Any]:
        """Dataset attrs as written by :meth:`to_netcdf`: ``ds.attrs`` plus the metadata, JSON-safe."""
        attrs = dict(self.ds.attrs)
        if self.metadata:
            attrs["metadata"] = to_json_safe(self.metadata)
        return attrs

    def to_csv(
        self,
        filepath: Union[str, Path],
        wavenumber_units: Literal["cm-1", "m-1"] = "cm-1",
    ) -> None:
        """Write the spectra to a CSV file (not for maps; use :meth:`to_netcdf`).

        The index column is the wavenumber (``raman_shift`` for Raman data);
        a series gets one column per scan, labelled ``tos_<seconds>s`` or
        ``scan_<n>``.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Destination file path; overwritten if it exists.
        wavenumber_units : {"cm-1", "m-1"}, optional
            Units of the index column (default is "cm-1").

        Raises
        ------
        ValueError
            For a map.
        """
        filepath = Path(filepath)
        if self.ndim > 2:
            raise ValueError("CSV export is not supported for map data (ndim > 2). Use to_netcdf() instead.")

        axis = self.wavenumber_per_cm if wavenumber_units == "cm-1" else self.wavenumber
        axis_name = "raman_shift" if self.technique == "raman" else "wavenumber"
        index = pd.Index(axis, name=f"{axis_name} [{wavenumber_units}]")
        if self.ndim == 1:
            df = pd.DataFrame({self.values_label: self.values}, index=index)
        else:
            tos = self.tos
            col_labels = (
                [f"tos_{t:.2f}s" for t in tos] if tos is not None
                else [f"scan_{i}" for i in range(self.ds.sizes["scan"])]
            )
            df = pd.DataFrame(self.values.T, index=index, columns=col_labels)
        df.to_csv(filepath)
        logger.debug("Saved CSV to %s", filepath)

    def to_netcdf(self, filepath: Union[str, Path]) -> None:
        """Save the dataset to a NetCDF file, history and all.

        Nested attributes (the :attr:`history`, lists of text, booleans,
        ``None``) and non-ASCII text are stored as JSON strings prefixed with
        ``"json:"``, so the file is readable by the NetCDF3 writer of scipy
        too; :meth:`from_netcdf` turns them back.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Destination file path.  An existing file will be overwritten with
            a warning.
        """
        filepath = Path(filepath)
        if filepath.exists():
            logger.warning(f"Overwriting existing file: {filepath}")
        ds = self.ds.copy()
        ds.attrs = encode_attrs(self._netcdf_attrs())
        for name in ds.variables:
            ds[name].attrs = encode_attrs(dict(ds[name].attrs))
        ds.to_netcdf(filepath, engine="h5netcdf")
        logger.debug(f"Saved NetCDF → {filepath}")

    def _spectrochempy_title(self) -> str:
        return _DATA_TYPE_TO_SCP_TITLE.get(self.data_type, str(self.data_type or "intensity"))

    def to_spectrochempy(self):
        """Convert to a spectrochempy ``NDDataset``; :meth:`from_spectrochempy` reads it back.

        The values become ``nd.data`` (always 2-D, ``(scan, wavenumber)``),
        titled with the OMNIC label of ``data_type`` (``'absorbance'``,
        ``'Kubelka-Munk'``, ...; Raman types as they are); the wavenumber
        ``nd.x`` in cm⁻¹; ``tos`` ``nd.y`` in seconds with the absolute
        timestamps as labels; the dataset attributes (``technique``,
        ``excitation_wavelength_nm``, ...) in ``nd.meta``; and the
        :attr:`history` in ``nd.history``. The background and the baseline
        are not carried.

        Returns
        -------
        spectrochempy.NDDataset
            The dataset.

        Raises
        ------
        ImportError
            If spectrochempy is not installed.
        ValueError
            For a map.
        """
        import spectrochempy as scp  # deferred: optional and slow to import

        if self.is_map:
            raise ValueError("to_spectrochempy supports a single spectrum or a series, not a map")
        values = np.atleast_2d(self.values)
        nd = scp.NDDataset(values, title=self._spectrochempy_title())
        x = scp.Coord(self.wavenumber_per_cm, title="wavenumber", units="1/cm")
        if self.tos is not None:
            timestamps = self.timestamps
            labels = [ts.isoformat() for ts in timestamps] if timestamps is not None else None
            y = scp.Coord(np.asarray(self.tos, dtype=float), title="time on stream", units="s", labels=labels)
        else:
            y = scp.Coord(np.arange(values.shape[0], dtype=float), title="scan")
        nd.set_coordset(y=y, x=x)
        for key, value in self.ds.attrs.items():
            if key != HISTORY_KEY:
                setattr(nd.meta, key, value)
        for line in self.format_history().splitlines():
            nd.history = line
        return nd

    def to_ramanspy(self):
        """Convert to a RamanSPy container; :meth:`from_ramanspy` reads it back.

        A single spectrum becomes ``ramanspy.Spectrum``, a series
        ``ramanspy.SpectralContainer`` (one spectrum per scan) and a map
        ``ramanspy.SpectralImage``; the spectral axis is the wavenumber (the
        Raman shift) in cm⁻¹. ``tos``, the history and the metadata stay
        here: RamanSPy containers have no place for them.

        Returns
        -------
        ramanspy.Spectrum, ramanspy.SpectralContainer or ramanspy.SpectralImage
            The container.

        Raises
        ------
        ImportError
            If ramanspy is not installed.
        """
        import ramanspy  # deferred: optional dependency

        values = np.asarray(self.values, dtype=float)
        axis = np.asarray(self.wavenumber_per_cm, dtype=float)
        if self.ndim == 1:
            return ramanspy.Spectrum(values, axis)
        if self.is_map:
            return ramanspy.SpectralImage(values, axis)
        return ramanspy.SpectralContainer(values, axis)

    @classmethod
    def _read_netcdf(cls, filepath: Union[str, Path]) -> xr.Dataset:
        """Read a file written by :meth:`to_netcdf` into memory, attributes decoded."""
        # read into memory and close: a lazy copy reopens the file later and keeps it locked (Windows)
        with xr.open_dataset(filepath) as handle:
            ds = handle.load()
        ds.attrs = decode_attrs(dict(ds.attrs))
        for name in ds.variables:
            ds[name].attrs = decode_attrs(dict(ds[name].attrs))
        return ds

    @classmethod
    @records_step
    def from_netcdf(cls, filepath: Union[str, Path]) -> "SpectroscopyData":
        """Load from a NetCDF file saved with :meth:`to_netcdf`.

        The stored :attr:`history` is restored and this load is appended to
        it. Files of the former DataArray-based Raman class (axis ``shift``)
        load too.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to the ``.nc`` file.

        Returns
        -------
        SpectroscopyData
            New instance reconstructed from the file.
        """
        return cls._from_netcdf_dataset(cls._read_netcdf(filepath))

    @classmethod
    def _from_netcdf_dataset(cls, ds: xr.Dataset) -> "SpectroscopyData":
        """Build from a decoded Dataset read by :meth:`from_netcdf`, restoring the metadata field."""
        if "data" not in ds:
            names = list(ds.data_vars)
            if len(names) != 1:
                raise ValueError(f"Expected a 'data' variable or exactly one data variable, found {names}")
            ds = _dataset_from_dataarray(ds[names[0]])
        attrs = dict(ds.attrs)
        metadata = attrs.pop("metadata", {})
        ds.attrs = attrs
        return cls(ds=ds, metadata=metadata if isinstance(metadata, dict) else {"metadata": metadata})

    @classmethod
    @records_step
    def from_xarray(
        cls,
        da: Union[xr.DataArray, xr.Dataset],
    ) -> "SpectroscopyData":
        """Construct from an existing xarray DataArray or Dataset.

        Parameters
        ----------
        da : xr.DataArray or xr.Dataset
            Source object.  A ``DataArray`` is wrapped in a ``Dataset`` under
            the key ``'data'``.  A ``Dataset`` is used directly.

        Returns
        -------
        SpectroscopyData
            New instance wrapping the provided xarray object.
        """
        if isinstance(da, xr.DataArray):
            ds = xr.Dataset({"data": da.copy()}, attrs=dict(da.attrs))
        else:
            ds = da.copy()
        return cls(ds=ds)

    @classmethod
    @records_step
    def from_spectrochempy(
        cls,
        nd,
        *,
        data_type: Optional[str] = None,
        wavenumber_to_si_factor: float = 100.0,
        delta_time_seconds: Optional[float] = None,
        tos_start: Optional[Union[pd.Timestamp, str]] = None,
        strict_tos_start: bool = True,
    ) -> "SpectroscopyData":
        """Construct from a spectrochempy ``NDDataset``: read there, processed there, or built by hand.

        The inverse of :meth:`to_spectrochempy`: ``nd.x`` is the wavenumber
        axis (made ascending), ``nd.data`` the values, ``nd.y`` timestamps
        give ``tos``. ``data_type`` comes from the title (``'absorbance'``,
        ``'Kubelka-Munk'``, ``'counts'``, ...) unless given; attributes
        stored in ``nd.meta`` by :meth:`to_spectrochempy` (``technique``,
        ``excitation_wavelength_nm``, ...) are restored, and a dataset
        without them gets the technique its ``data_type`` belongs to.

        Parameters
        ----------
        nd : spectrochempy.NDDataset
            Source dataset.
        data_type : str or None, optional
            Overrides what the title says.
        wavenumber_to_si_factor : float, optional
            Multiplicative factor converting the dataset's x-axis to m⁻¹
            (default is ``100.0``, for cm⁻¹).
        delta_time_seconds : float or None, optional
            Fixed time step between scans when ``nd.y`` carries no
            timestamps.  Mutually exclusive with ``tos_start``.
        tos_start : pandas.Timestamp or str or None, optional
            Origin of ``tos`` (default: the first timestamp).
        strict_tos_start : bool, optional
            Raise when the timestamp found in the dataset cannot be parsed
            (default is ``True``); otherwise warn and continue without one.

        Returns
        -------
        SpectroscopyData
            New 2-D ``(scan, wavenumber)`` instance, also for a single
            spectrum: one scan is the one-element case of a series.

        Raises
        ------
        ValueError
            If both ``delta_time_seconds`` and ``tos_start`` are given, if
            the title names no known ``data_type`` and none is given, or if
            ``strict_tos_start`` is ``True`` and the timestamp cannot be
            parsed.
        """
        from phd_parser.spectroscopy.infrared import spectrochempy as _scp_parser

        if delta_time_seconds is not None and tos_start is not None:
            raise ValueError("Specify either 'delta_time_seconds' or 'tos_start', not both.")

        raw = _scp_parser.read_nddataset(nd, delta_time_seconds=delta_time_seconds, tos_start=tos_start)
        meta: dict[str, Any] = {}
        try:
            meta = {key: value for key, value in dict(nd.meta.data).items() if key not in ("merge_log",)}
        except Exception:  # noqa: BLE001  a dataset without our meta
            meta = {}

        title = raw["meta"].get("vlabel", "")
        resolved = data_type or _data_type_from_title(title) or meta.get("data_type")
        if resolved is None:
            raise ValueError(
                f"spectrochempy dataset title '{title}' has no known data_type mapping. "
                f"Known titles: {list(_OMNIC_VLABEL_TO_DATA_TYPE)} and the data_type names "
                f"{list(DataType.__args__)}. Pass data_type=... explicitly."
            )
        attrs = {key: value for key, value in meta.items() if key not in ("tos_start", "wavenumber_unit", "tos_unit")}
        attrs["data_type"] = resolved
        if attrs.get("technique") is None:
            technique = _technique_of_data_type(resolved)
            if technique is not None:
                attrs["technique"] = technique

        parsed_tos_start = cls._parse_tos_start(tos_start, raw["meta"].get("tos_start"), strict_tos_start)
        if parsed_tos_start is not None:
            attrs["tos_start"] = parsed_tos_start.isoformat()

        ds = cls._build_ds(
            np.asarray(raw["data"]["x"], dtype=float) * wavenumber_to_si_factor,
            np.asarray(raw["data"]["v"], dtype=float),
            tos=np.asarray(raw["data"]["tos"], dtype=float) if "tos" in raw["data"] else None,
            attrs=attrs,
        )
        return cls(ds=ds)

    @staticmethod
    def _parse_tos_start(
        given: Optional[Union[pd.Timestamp, str]],
        found: Any,
        strict: bool,
    ) -> Optional[pd.Timestamp]:
        """The origin a reader uses: the one given by the caller, else the one found in the file."""
        if given is not None:
            return pd.Timestamp(given)
        if found is None:
            return None
        try:
            return pd.Timestamp(found)
        except Exception as exc:  # noqa: BLE001  anything pandas cannot read
            if strict:
                raise ValueError(f"Could not parse tos_start '{found}': {exc}") from exc
            logger.warning("Ignoring unparseable tos_start '%s': %s", found, exc)
            return None

    @classmethod
    def _from_raman_parts(
        cls,
        shift_si: npt.NDArray,
        values: npt.NDArray,
        excitation_wavelength_nm: Optional[float],
        data_type: str,
        tos: Optional[npt.NDArray] = None,
        tos_start: Optional[Union[pd.Timestamp, str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        map_coords: Optional[dict[str, npt.NDArray]] = None,
    ) -> "SpectroscopyData":
        """Build a Raman instance from SI arrays (shared by the Raman readers; maps included)."""
        attrs: dict[str, Any] = {"technique": "raman", "data_type": data_type}
        if excitation_wavelength_nm is not None:
            attrs["excitation_wavelength_nm"] = float(excitation_wavelength_nm)
        if tos_start is not None:
            attrs["tos_start"] = pd.Timestamp(tos_start).isoformat()
        ds = cls._build_ds(shift_si, values, tos=tos, attrs=attrs)
        if map_coords:
            ds = ds.assign_coords(map_coords)
        return cls(ds=ds, metadata=metadata or {})

    @classmethod
    @records_step
    def from_ramanspy(
        cls,
        container,
        *,
        excitation_wavelength_nm: Optional[float] = None,
        data_type: str = "intensity",
        tos: Optional[npt.NDArray] = None,
        tos_start: Optional[Union[pd.Timestamp, str]] = None,
    ) -> "SpectroscopyData":
        """Construct from a RamanSPy ``Spectrum``, ``SpectralContainer`` or ``SpectralImage``.

        The inverse of :meth:`to_ramanspy`, e.g. after preprocessing there.
        ``spectral_axis`` is the Raman shift in cm⁻¹. A 1-D container gives a
        single spectrum, a 2-D one a series (pass ``tos`` to time it), an
        image a ``(y, x, wavenumber)`` map. RamanSPy keeps no excitation
        wavelength, so pass it to get the scattered-light axes.

        Parameters
        ----------
        container : ramanspy.Spectrum, SpectralContainer or SpectralImage
            Anything with ``spectral_data`` and ``spectral_axis``.
        excitation_wavelength_nm : float, optional
            Excitation laser wavelength in nm.
        data_type : {"intensity", "counts", "counts_per_second", "arbitrary"}, optional
            What the values are (default "intensity").
        tos : numpy.ndarray, optional
            Elapsed seconds per spectrum of a series.
        tos_start : pandas.Timestamp or str, optional
            Absolute start time.

        Returns
        -------
        SpectroscopyData
            New instance with ``technique = 'raman'``.

        Raises
        ------
        ValueError
            For containers of more than three dimensions, or a ``tos`` that
            does not match.
        """
        values = np.asarray(container.spectral_data, dtype=float)
        shift_si = np.asarray(container.spectral_axis, dtype=float) * 100.0
        if values.shape[-1] != shift_si.size:
            raise ValueError(f"spectral_data.shape[-1] ({values.shape[-1]}) != spectral_axis size ({shift_si.size})")
        map_coords = None
        if values.ndim == 3:
            map_coords = {"y": np.arange(values.shape[0]), "x": np.arange(values.shape[1])}
            tos = None
        elif values.ndim == 2 and tos is not None:
            tos = np.asarray(tos, dtype=float)
            if tos.size != values.shape[0]:
                raise ValueError(f"tos size ({tos.size}) != number of spectra ({values.shape[0]})")
        elif values.ndim > 3:
            raise ValueError(f"SpectroscopyData holds 1-D, 2-D or (y, x, wavenumber) data, got shape {values.shape}")
        return cls._from_raman_parts(shift_si, values, excitation_wavelength_nm, data_type,
                                     tos=tos, tos_start=tos_start, map_coords=map_coords)
