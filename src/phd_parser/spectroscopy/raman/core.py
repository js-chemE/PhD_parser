"""Raman spectra: :class:`RamanData`, a :class:`~phd_parser.spectroscopy.SpectroscopyData` with the Raman readers.

The axis is the Raman shift, which is the vibrational wavenumber: it is
stored as ``ds.coords["wavenumber"]`` (m⁻¹, Stokes positive) like an IR
spectrum, so IR and Raman of the same sample line up band for band.
``shift`` / ``shift_per_cm`` and ``select_shift_range`` are the Raman names
for the same thing. The excitation wavelength is
``ds.attrs["excitation_wavelength_nm"]``; the absolute wavenumber of the
scattered light is ``scattered_wavenumber``.
"""

import logging
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pydantic import Field, model_validator
from scipy import constants as const

from phd_parser.history import records_step, to_json_safe
from phd_parser.spectroscopy.core import SpectroscopyData
from phd_parser.spectroscopy.raman import btc655n, renishaw

logger = logging.getLogger(__name__)

VLabel = Literal["intensity", "counts", "counts_per_second", "arbitrary"]


class RamanData(SpectroscopyData):
    """Raman spectra: a :class:`SpectroscopyData` read from Raman instruments.

    ``ds["data"]`` runs over ``wavenumber`` (the Raman shift, m⁻¹, Stokes
    positive) as ``(wavenumber,)``, ``(scan, wavenumber)`` with ``tos`` or
    ``(y, x, wavenumber)`` for a map. ``ds.attrs`` hold ``technique``
    (``'raman'``), ``excitation_wavelength_nm`` and ``data_type`` (what the
    values are: ``intensity``, ``counts``, ``counts_per_second`` or
    ``arbitrary``, also readable as :attr:`values_label`). The raw metadata
    of the source file is the separate field :attr:`metadata`.

    ``RamanData(da=..., excitation_wavelength_nm=..., values_label=...)``,
    the constructor of the former DataArray-based class, still works, and so
    do its NetCDF files (axis ``shift``).
    """

    _technique: ClassVar[Optional[str]] = "raman"
    _default_data_type: ClassVar[Optional[str]] = "intensity"

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw metadata extracted from the source file.",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_dataarray(cls, data: Any) -> Any:
        """Accept the keywords of the former DataArray-based class (``da``, ``excitation_wavelength_nm``, ``values_label``)."""
        if isinstance(data, dict) and "da" in data and "ds" not in data:
            data = dict(data)
            data["ds"] = _dataset_from_dataarray(
                data.pop("da"),
                excitation_wavelength_nm=data.pop("excitation_wavelength_nm", None),
                values_label=data.pop("values_label", None),
            )
        return data

    # ----------------------------------------------------------------
    # Raman names
    # ----------------------------------------------------------------

    @property
    def shift(self) -> npt.NDArray:
        """Raman shift in m⁻¹, Stokes positive (the ``wavenumber`` axis)."""
        return self.wavenumber

    @property
    def shift_per_cm(self) -> npt.NDArray:
        """Raman shift in cm⁻¹, Stokes positive."""
        return self.wavenumber_per_cm

    @property
    def n_spectral(self) -> int:
        """Number of points on the shift axis."""
        return self.ds.sizes["wavenumber"]

    @property
    def values_label(self) -> str:
        """What the values are (``intensity``, ``counts``, ...): the ``data_type``."""
        return str(self.data_type or "intensity")

    @property
    def da(self) -> xr.DataArray:
        """The spectra as a DataArray (``ds["data"]``), named after :attr:`values_label`."""
        return self.ds["data"].rename(self.values_label)

    @property
    def excitation_wavelength_nm(self) -> Optional[float]:
        """Excitation laser wavelength in nm (``ds.attrs['excitation_wavelength_nm']``)."""
        value = self.ds.attrs.get("excitation_wavelength_nm")
        return None if value is None else float(value)

    def _require_excitation(self) -> float:
        if self.excitation_wavelength_nm is None:
            raise ValueError("No excitation_wavelength_nm stored; the scattered axes cannot be computed.")
        return self.excitation_wavelength_nm

    @property
    def excitation_wavenumber_per_cm(self) -> float:
        """Excitation laser wavenumber in cm⁻¹."""
        return 1e7 / self._require_excitation()

    @property
    def excitation_wavenumber(self) -> float:
        """Excitation laser wavenumber in m⁻¹."""
        return self.excitation_wavenumber_per_cm * 100.0

    @property
    def scattered_wavenumber(self) -> npt.NDArray:
        """Absolute wavenumber of the scattered light in m⁻¹ (Stokes: excitation minus shift)."""
        return self.excitation_wavenumber - self.shift

    @property
    def scattered_wavenumber_per_cm(self) -> npt.NDArray:
        """Absolute wavenumber of the scattered light in cm⁻¹."""
        return self.scattered_wavenumber / 100.0

    @property
    def scattered_wavelength_nm(self) -> npt.NDArray:
        """Wavelength of the scattered light in nm (what a spectrometer without shift calibration shows)."""
        return 1e9 / self.scattered_wavenumber

    @property
    def scattered_frequency(self) -> npt.NDArray:
        """Frequency of the scattered light in Hz."""
        return self.scattered_wavenumber * const.c

    def get_evolution(
        self,
        shift_per_cm: Union[float, list[float], npt.NDArray],
        method: Literal["nearest", "linear"] = "nearest",
        tolerance_per_cm: Optional[float] = None,
        rolling_window: Optional[int] = None,
    ) -> xr.DataArray:
        """Return intensity vs. scan at one or more Raman shifts (cm⁻¹).

        :meth:`SpectroscopyData.get_evolution` under its Raman argument name.

        Returns
        -------
        xarray.DataArray
            Dims ``('scan',)`` for a scalar target, ``('scan', 'wavenumber')``
            for a sequence.
        """
        return super().get_evolution(shift_per_cm, method, tolerance_per_cm, rolling_window)

    @records_step
    def sort(self, ascending: bool = True, by: Optional[Union[str, Sequence[str]]] = None) -> "RamanData":
        """Return a new RamanData sorted along the shift axis (or ``by``).

        Parameters
        ----------
        ascending : bool, optional
            Sort ascending when ``True`` (default).
        by : str or sequence of str, optional
            Coordinate(s) to sort by (default: the shift axis).
        """
        return super().sort(by=by if by is not None else "wavenumber", ascending=ascending)

    @records_step
    def select_shift_range(
        self,
        min_shift_per_cm: Optional[float] = None,
        max_shift_per_cm: Optional[float] = None,
    ) -> "RamanData":
        """Return a new RamanData restricted to a Raman shift range in cm⁻¹ (inclusive).

        :meth:`SpectroscopyData.select_wavenumber_range` under its Raman name.
        """
        return self.select_wavenumber_range(min_shift_per_cm, max_shift_per_cm)

    def _repr_extra(self) -> str:
        nm = self.excitation_wavelength_nm
        return f", excitation={nm:.1f} nm" if nm is not None else ""

    # ----------------------------------------------------------------
    # Export
    # ----------------------------------------------------------------

    def to_csv(
        self,
        filepath: Union[str, Path],
        shift_units: Literal["cm-1", "m-1"] = "cm-1",
    ) -> None:
        """Write the spectra to a CSV file (not for maps; use :meth:`to_netcdf`).

        Parameters
        ----------
        filepath : str or pathlib.Path
            Destination file path; overwritten if it exists.
        shift_units : {"cm-1", "m-1"}, optional
            Units of the shift index column (default is "cm-1").
        """
        filepath = Path(filepath)
        if self.ndim > 2:
            raise ValueError("CSV export is not supported for map data (ndim > 2). Use to_netcdf() instead.")

        shift = self.shift_per_cm if shift_units == "cm-1" else self.shift
        shift_label = f"raman_shift [{shift_units}]"
        if self.ndim == 1:
            df = pd.DataFrame({self.values_label: self.values}, index=pd.Index(shift, name=shift_label))
        else:
            tos = self.tos
            col_labels = (
                [f"tos_{t:.2f}s" for t in tos] if tos is not None
                else [f"scan_{i}" for i in range(self.ds.sizes["scan"])]
            )
            df = pd.DataFrame(self.values.T, index=pd.Index(shift, name=shift_label), columns=col_labels)
        df.to_csv(filepath)
        logger.debug("Saved CSV → %s", filepath)

    def _netcdf_attrs(self) -> dict[str, Any]:
        attrs = dict(self.ds.attrs)
        if self.metadata:
            attrs["metadata"] = to_json_safe(self.metadata)
        return attrs

    def to_ramanspy(self):
        """Convert to a RamanSPy container; :meth:`from_ramanspy` reads it back.

        A single spectrum becomes ``ramanspy.Spectrum``, a series
        ``ramanspy.SpectralContainer`` (one spectrum per scan) and a map
        ``ramanspy.SpectralImage``; the spectral axis is the Raman shift in
        cm⁻¹. ``tos``, the history and the metadata stay here: RamanSPy
        containers have no place for them.

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
        axis = np.asarray(self.shift_per_cm, dtype=float)
        if self.ndim == 1:
            return ramanspy.Spectrum(values, axis)
        if self.is_map:
            return ramanspy.SpectralImage(values, axis)
        return ramanspy.SpectralContainer(values, axis)

    # ----------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------

    @classmethod
    def _from_parts(
        cls,
        shift_si: npt.NDArray,
        values: npt.NDArray,
        excitation_wavelength_nm: Optional[float],
        values_label: str,
        tos: Optional[npt.NDArray] = None,
        tos_start: Optional[Union[pd.Timestamp, str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        map_coords: Optional[dict[str, npt.NDArray]] = None,
    ) -> "RamanData":
        attrs: dict[str, Any] = {"data_type": values_label}
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
    def from_arrays(
        cls,
        shift_per_cm: npt.NDArray,
        values: npt.NDArray,
        excitation_wavelength_nm: float,
        values_label: VLabel = "intensity",
        tos: Optional[npt.NDArray] = None,
        metadata: Optional[dict[str, Any]] = None,
        tos_start: Optional[Union[pd.Timestamp, str]] = None,
    ) -> "RamanData":
        """Construct a RamanData from raw NumPy arrays.

        Parameters
        ----------
        shift_per_cm : numpy.ndarray
            1-D array of Raman shift values in cm⁻¹.
        values : numpy.ndarray
            1-D (single spectrum) or 2-D ``(n_scans, n_spectral)`` array; the
            last axis must match ``shift_per_cm``.
        excitation_wavelength_nm : float
            Excitation laser wavelength in nm.
        values_label : {"intensity", "counts", "counts_per_second", "arbitrary"}, optional
            What the values are, stored as ``data_type`` (default "intensity").
        tos : numpy.ndarray, optional
            Elapsed seconds per scan; ignored for 1-D ``values``.
        metadata : dict, optional
            Provenance metadata stored on the instance.
        tos_start : pandas.Timestamp or str, optional
            Absolute start time of the measurement.

        Returns
        -------
        RamanData
            New instance built from the provided arrays.

        Raises
        ------
        ValueError
            When the shapes do not match or the excitation wavelength is not positive.
        """
        shift_si = np.asarray(shift_per_cm, dtype=float) * 100.0
        values = np.asarray(values, dtype=float)

        if shift_si.ndim != 1:
            raise ValueError("shift_per_cm must be 1-D")
        if values.ndim not in (1, 2):
            raise ValueError(f"values must be 1-D or 2-D, got shape {values.shape}")
        if values.shape[-1] != shift_si.size:
            raise ValueError(f"values.shape[-1] ({values.shape[-1]}) != shift size ({shift_si.size})")
        if float(excitation_wavelength_nm) <= 0:
            raise ValueError(f"excitation_wavelength_nm must be positive, got {excitation_wavelength_nm}")
        if values.ndim == 1:
            tos = None
        elif tos is not None:
            tos = np.asarray(tos, dtype=float)
            if tos.size != values.shape[0]:
                raise ValueError(f"tos size ({tos.size}) != values.shape[0] ({values.shape[0]})")

        return cls._from_parts(shift_si, values, excitation_wavelength_nm, values_label,
                               tos=tos, tos_start=tos_start, metadata=metadata)

    @classmethod
    @records_step
    def from_ramanspy(
        cls,
        container,
        excitation_wavelength_nm: Optional[float] = None,
        values_label: VLabel = "intensity",
        tos: Optional[npt.NDArray] = None,
        tos_start: Optional[Union[pd.Timestamp, str]] = None,
    ) -> "RamanData":
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
        values_label : {"intensity", "counts", "counts_per_second", "arbitrary"}, optional
            What the values are (default "intensity").
        tos : numpy.ndarray, optional
            Elapsed seconds per spectrum of a series.
        tos_start : pandas.Timestamp or str, optional
            Absolute start time.

        Returns
        -------
        RamanData
            New instance.

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
            raise ValueError(f"RamanData holds 1-D, 2-D or (y, x, shift) data, got shape {values.shape}")
        return cls._from_parts(shift_si, values, excitation_wavelength_nm, values_label,
                               tos=tos, tos_start=tos_start, map_coords=map_coords)

    @classmethod
    def _from_netcdf_dataset(cls, ds: xr.Dataset) -> "RamanData":
        if "data" not in ds:
            names = list(ds.data_vars)
            if len(names) != 1:
                raise ValueError(f"Expected one data variable, found {names}")
            ds = _dataset_from_dataarray(ds[names[0]])
        if ds.attrs.get("excitation_wavelength_nm") is None:
            raise ValueError(
                "NetCDF file is missing 'excitation_wavelength_nm' attribute. "
                "Provide it manually via from_arrays()."
            )
        attrs = dict(ds.attrs)
        metadata = attrs.pop("metadata", {})
        ds.attrs = attrs
        return cls(ds=ds, metadata=metadata if isinstance(metadata, dict) else {"metadata": metadata})

    @classmethod
    @records_step
    def from_btc655n_export(
        cls,
        filepath: Union[str, Path],
        y_key: btc655n.Y_KEYS = "Raw data #1",
        remove_empty: bool = True,
    ) -> "RamanData":
        """Load a RamanData from a B&W Tek BTC655N spectrometer export file.

        The BTC655N exports wavelength in nm; this constructor converts those
        values to Raman shift in cm⁻¹ using the excitation wavelength found
        in the file metadata.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to the BTC655N text export file.
        y_key : str, optional
            Column to use as the values (default "Raw data #1"), one of
            :data:`phd_parser.spectroscopy.raman.btc655n.Y_KEYS`.
        remove_empty : bool, optional
            Drop rows with no wavelength data when ``True`` (default).

        Raises
        ------
        ValueError
            When the export metadata does not contain ``laser_wavelength``.
        """
        raw = btc655n.read_export(filepath, remove_empty=remove_empty)

        excitation_nm = raw["meta"].get("laser_wavelength")
        if excitation_nm is None:
            raise ValueError(
                "BTC655N export is missing 'laser_wavelength' in metadata. "
                "Cannot compute Raman shift without excitation wavelength."
            )

        # BTC655N reports wavelength in nm → convert to Raman shift in cm⁻¹
        wavelength_nm = np.asarray(raw["data"]["Wavelength"], dtype=float)
        wavenumber_per_cm = 1e7 / wavelength_nm                             # scattered (cm⁻¹)
        excitation_per_cm = 1e7 / float(excitation_nm)                      # excitation (cm⁻¹)
        shift_per_cm = excitation_per_cm - wavenumber_per_cm                # Stokes shift (cm⁻¹)

        return cls.from_arrays(
            shift_per_cm=shift_per_cm,
            values=np.asarray(raw["data"][y_key], dtype=float),
            excitation_wavelength_nm=float(excitation_nm),
            metadata=raw["meta"],
        )

    @classmethod
    @records_step
    def from_renishaw_txt(
        cls,
        filepath: Union[str, Path],
        excitation_wavelength_nm: float,
    ) -> "RamanData":
        """Load a RamanData from a Renishaw WiRe plain-text export (shift already in cm⁻¹).

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to the Renishaw ``.txt`` export file.
        excitation_wavelength_nm : float
            Excitation laser wavelength in nm (not in the file).
        """
        raw = renishaw.read_export_txt(Path(filepath))
        shift_per_cm = np.asarray(raw["data"]["wavenumber"], dtype=float)
        return cls.from_arrays(
            shift_per_cm=shift_per_cm,
            values=np.asarray(raw["data"]["intensity"], dtype=float),
            excitation_wavelength_nm=excitation_wavelength_nm,
            metadata=raw["meta"],
        )

    @classmethod
    @records_step
    def from_renishaw_wdf(cls, filepath: Union[str, Path]) -> "RamanData":
        """Load a RamanData from a Renishaw WiRe binary WDF file.

        The excitation wavelength and shift axis come from the WDF header.
        Single spectra, time series (2-D) and maps (``(y, x, wavenumber)``)
        are supported.

        Raises
        ------
        ValueError
            When the x unit is not ``RamanShift``, the data unit not
            ``Counts``, or the axis length does not match the data.
        """
        filepath = Path(filepath)
        wdf = renishaw.read_export_wdf(filepath)

        if wdf.x_unit != renishaw.UnitType.RamanShift:
            raise ValueError(
                f"Expected x_unit RamanShift, got {wdf.x_unit}. "
                "Cannot interpret data as Raman shift."
            )
        if wdf.data_unit != renishaw.UnitType.Counts:
            raise ValueError(f"Expected data_unit Counts, got {wdf.data_unit}.")

        # WDF stores the shift axis directly in cm⁻¹
        shift_per_cm = np.asarray(wdf.wavenumber, dtype=float)
        shift_si = shift_per_cm * 100.0
        # Excitation wavelength from WDF header (laser_cm1 is in cm⁻¹)
        excitation_nm = 1e7 / float(wdf.laser_cm1)
        data = np.asarray(wdf.data, dtype=float)

        if data.ndim == 3:  # map, WDF order (n_y, n_x, n_spectral)
            n_y, n_x, n_spec = data.shape
            if n_spec != shift_si.size:
                raise ValueError(
                    f"WDF spectral axis length ({shift_si.size}) does not match data.shape[-1] ({n_spec})"
                )
            return cls._from_parts(
                shift_si, data, excitation_nm, "counts",
                metadata={"WDFResult": wdf},
                map_coords={"x": np.arange(n_x), "y": np.arange(n_y)},
            )

        return cls.from_arrays(
            shift_per_cm=shift_per_cm,
            values=data,
            excitation_wavelength_nm=excitation_nm,
            values_label="counts",
            metadata={"WDFResult": wdf},
        )


def _dataset_from_dataarray(
    da: xr.DataArray,
    excitation_wavelength_nm: Optional[float] = None,
    values_label: Optional[str] = None,
) -> xr.Dataset:
    """Turn the DataArray of the former ``RamanData`` (axis ``shift``) into the current Dataset layout."""
    if not isinstance(da, xr.DataArray):
        raise TypeError(f"'da' must be an xr.DataArray, got {type(da)}")
    attrs = dict(da.attrs)
    label = values_label or attrs.pop("values_label", None) or "intensity"
    attrs.pop("values_label", None)
    attrs.pop("shift_units", None)
    excitation = excitation_wavelength_nm if excitation_wavelength_nm is not None else attrs.get("excitation_wavelength_nm")
    if excitation is not None:
        attrs["excitation_wavelength_nm"] = float(excitation)
    attrs.update({"data_type": label, "technique": "raman", "wavenumber_unit": "m^-1"})
    data = da.rename("data")
    if "shift" in data.dims:
        data = data.rename({"shift": "wavenumber"})
    data.attrs = {}
    return xr.Dataset({"data": data}, attrs=attrs)
