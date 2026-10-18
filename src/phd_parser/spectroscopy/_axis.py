"""The spectral axis in every unit, the Raman names, and the getters of scans, baselines and evolutions.

A mixin of :class:`~phd_parser.spectroscopy.core.SpectroscopyData`; never used on its own.
"""

import logging
from typing import TYPE_CHECKING
from functools import cached_property
from typing import Literal, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy import constants as const

from phd_parser.spectroscopy._common import (
    _MAP_DIMS,
    DataType,
)
from phd_parser.spectroscopy.units import axis_label as _axis_label, from_wavenumber, quantity_of

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class _AxisMixin:
    """The spectral axis in every unit, the Raman names, and the getters of scans, baselines and evolutions."""

    @property
    def technique(self) -> Optional[str]:
        """Which technique recorded the data (``'infrared'``, ``'raman'`` or ``None``), from ``ds.attrs['technique']``."""
        return self.ds.attrs.get("technique")

    @property
    def is_map(self) -> bool:
        """Whether the data are a spatial map ``(y, x, wavenumber)``."""
        return all(dim in self.ds["data"].dims for dim in _MAP_DIMS)

    def get_map_spectrum(self, x: int, y: int) -> npt.NDArray:
        """Return the spectrum at one map pixel.

        Parameters
        ----------
        x : int
            Zero-based index along the ``x`` dimension.
        y : int
            Zero-based index along the ``y`` dimension.

        Returns
        -------
        numpy.ndarray
            1-D array of length ``n_wavenumber``.

        Raises
        ------
        ValueError
            When the data are not a map.
        """
        if not self.is_map:
            raise ValueError("get_map_spectrum requires 'x' and 'y' dimensions")
        return self.ds["data"].isel(x=x, y=y).values

    @property
    def ndim(self) -> int:
        """Number of dimensions of the data variable (1 or 2).

        Returns
        -------
        int
            ``1`` for a single spectrum, ``2`` for a time series
            ``(scan, wavenumber)``.
        """
        return self.ds["data"].ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the data variable.

        Returns
        -------
        tuple of int
            ``(n_wavenumber,)`` for 1-D data or
            ``(n_scan, n_wavenumber)`` for 2-D data.
        """
        return tuple(self.ds["data"].shape)

    @property
    def values(self) -> npt.NDArray:
        """Raw spectral intensity values as a NumPy array.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_wavenumber,)`` or ``(n_scan, n_wavenumber)``.
        """
        return self.ds["data"].values

    @property
    def wavenumber(self) -> npt.NDArray:
        """Wavenumber axis in SI units (m⁻¹).

        Returns
        -------
        numpy.ndarray
            1-D array of wavenumber values in m⁻¹.
        """
        # SI units (m⁻¹)
        return self.ds.coords["wavenumber"].values

    @property
    def wavenumber_per_cm(self) -> npt.NDArray:
        """Wavenumber axis in cm⁻¹.

        Returns
        -------
        numpy.ndarray
            1-D array of wavenumber values in cm⁻¹.
        """
        return self.wavenumber / 100.0

    @property
    def n_spectral(self) -> int:
        """Number of points on the wavenumber axis."""
        return self.ds.sizes["wavenumber"]

    def axis(self, unit: str = "cm^-1") -> npt.NDArray:
        """The spectral axis read in any unit: wavenumber, wavelength, frequency or photon energy.

        The axis is stored once, as a wavenumber in m⁻¹ (for Raman data the
        Raman shift); this converts it. A wavenumber, frequency or energy
        unit of a shift reads the shift itself (``'raman shift / cm^-1'``,
        ``'raman shift / meV'``). A wavelength unit of a shift reads the
        scattered light, ``1 / (excitation - shift)``, so it needs the
        excitation wavelength, as a spectrometer's wavelength axis would.

        Parameters
        ----------
        unit : str, optional
            Any of :data:`phd_parser.spectroscopy.units.SPECTRAL_UNITS` or a
            common spelling of it: ``'cm^-1'`` (default), ``'m^-1'``,
            ``'um'``, ``'nm'``, ``'A'``, ``'THz'``, ``'eV'``, ``'meV'``,
            ``'keV'``, ``'kJ mol^-1'``, ``'kcal mol^-1'``, ...

        Returns
        -------
        numpy.ndarray
            1-D array in ``unit``.

        Raises
        ------
        ValueError
            For an unknown unit, or a wavelength unit on a Raman shift axis
            without an excitation wavelength.
        """
        return from_wavenumber(self.wavenumber, unit, self._shift_zero_si(unit))

    def axis_label(self, unit: str = "cm^-1") -> str:
        """The plot label of :meth:`axis` in ``unit``, e.g. ``'wavenumber / cm^-1'`` or ``'raman shift / meV'``."""
        return _axis_label(unit, shift=self.excitation_wavelength_nm is not None)

    def _shift_zero_si(self, unit: str) -> Optional[float]:
        """The excitation wavenumber in m⁻¹ when ``unit`` reads the scattered light of a shift axis, else None."""
        if quantity_of(unit) != "wavelength" or self.excitation_wavelength_nm is None:
            if quantity_of(unit) == "wavelength" and self.technique == "raman":
                raise ValueError(
                    "A wavelength unit of a Raman shift reads the scattered light, which needs "
                    "excitation_wavelength_nm; this spectrum has none."
                )
            return None
        return self.excitation_wavenumber

    @property
    def shift(self) -> npt.NDArray:
        """Raman shift in m⁻¹, Stokes positive (the ``wavenumber`` axis under its Raman name)."""
        return self.wavenumber

    @property
    def shift_per_cm(self) -> npt.NDArray:
        """Raman shift in cm⁻¹, Stokes positive."""
        return self.wavenumber_per_cm

    @property
    def values_label(self) -> str:
        """What the values are (``intensity``, ``counts``, ``absorbance``, ...): the ``data_type``, or ``'data'`` if unset."""
        return str(self.data_type) if self.data_type else "data"

    @property
    def da(self) -> xr.DataArray:
        """The spectra as a DataArray (``ds["data"]``), named after :attr:`values_label`."""
        return self.ds["data"].rename(self.values_label)

    @property
    def excitation_wavelength_nm(self) -> Optional[float]:
        """Excitation laser wavelength in nm (``ds.attrs['excitation_wavelength_nm']``), or ``None``."""
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

    @property
    def tos(self) -> Optional[npt.NDArray]:
        """Elapsed time of each scan in seconds since ``tos_start``.

        Returns
        -------
        numpy.ndarray or None
            1-D array of elapsed seconds, or ``None`` if the ``'tos'``
            coordinate is absent.
        """
        # Elapsed seconds since first scan
        if "tos" in self.ds.coords:
            return self.ds.coords["tos"].values
        return None

    @property
    def data_type(self) -> Optional[DataType]:
        """Spectral quantity stored in ``ds['data']``.

        Returns
        -------
        str or None
            One of ``'single_beam'``, ``'absorbance'``, ``'transmittance'``,
            ``'reflectance'``, ``'log_1_r'``, ``'kubelka_munk'``, or ``None``
            if not set.
        """
        return self.ds.attrs.get("data_type")

    @property
    def has_background(self) -> bool:
        """Whether a background spectrum is stored in the dataset.

        Returns
        -------
        bool
            ``True`` if ``ds['background']`` exists.
        """
        return "background" in self.ds

    @property
    def background(self) -> Optional[npt.NDArray]:
        """Background spectrum values (single_beam, 1-D).

        Returns
        -------
        numpy.ndarray or None
            1-D array of background intensities, or ``None`` if no background
            has been assigned.
        """
        if "background" not in self.ds:
            return None
        return self.ds["background"].values

    @property
    def background_data_type(self) -> Optional[DataType]:
        """Data type label of the stored background variable.

        Returns
        -------
        str or None
            Value of ``ds['background'].attrs['data_type']``, or ``None`` if
            no background is present.
        """
        if "background" not in self.ds:
            return None
        return self.ds["background"].attrs.get("data_type")

    @property
    def has_baseline(self) -> bool:
        """Whether a baseline curve is stored in the dataset.

        Returns
        -------
        bool
            ``True`` if ``ds['baseline']`` exists.
        """
        return "baseline" in self.ds

    @property
    def baseline(self) -> Optional[npt.NDArray]:
        """Stored baseline curve, in the units of the current ``data_type``.

        Returns
        -------
        numpy.ndarray or None
            Array with the same shape as ``values``, or ``None`` if no
            baseline has been subtracted.
        """
        if "baseline" not in self.ds:
            return None
        return self.ds["baseline"].values

    @property
    def data_unbaselined(self) -> npt.NDArray:
        """Spectral values before baseline subtraction (``data + baseline``).

        Returns
        -------
        numpy.ndarray
            Array with the same shape as ``values``.  Equal to ``values``
            unchanged if no baseline has been subtracted.
        """
        if not self.has_baseline:
            return self.values.copy()
        return self.values + self.baseline

    @cached_property
    def wavelength(self) -> npt.NDArray:
        """Wavelength axis in metres (``1 / wavenumber``).

        Returns
        -------
        numpy.ndarray
            1-D array of wavelength values in m.
        """
        # metres
        return 1.0 / self.wavenumber

    @cached_property
    def wavelength_nm(self) -> npt.NDArray:
        """Wavelength axis in nanometres.

        Returns
        -------
        numpy.ndarray
            1-D array of wavelength values in nm.
        """
        return self.wavelength * 1e9

    @cached_property
    def wavelength_mum(self) -> npt.NDArray:
        """Wavelength axis in micrometres.

        Returns
        -------
        numpy.ndarray
            1-D array of wavelength values in µm.
        """
        return self.wavelength * 1e6

    @cached_property
    def frequency(self) -> npt.NDArray:
        """Frequency axis in hertz (``wavenumber * c``).

        Returns
        -------
        numpy.ndarray
            1-D array of frequency values in Hz.
        """
        # Hz
        return self.wavenumber * const.c

    @cached_property
    def energy(self) -> npt.NDArray:
        """Photon energy axis in joules (``wavenumber * h * c``).

        Returns
        -------
        numpy.ndarray
            1-D array of photon energies in J.
        """
        # Joules
        return self.wavenumber * const.Planck * const.c

    @cached_property
    def energy_eV(self) -> npt.NDArray:
        """Photon energy axis in electronvolts.

        Returns
        -------
        numpy.ndarray
            1-D array of photon energies in eV.
        """
        return self.energy / const.electron_volt

    @cached_property
    def energy_kJ_per_mol(self) -> npt.NDArray:
        """Molar photon energy axis in kJ mol⁻¹.

        Returns
        -------
        numpy.ndarray
            1-D array of molar photon energies in kJ mol⁻¹.
        """
        return 1e-3 * self.energy * const.Avogadro  # kJ mol^-1

    def get_scan(self, scan_index: int) -> npt.NDArray:
        """Return the intensity values of a single scan by integer index.

        Parameters
        ----------
        scan_index : int
            Zero-based index of the scan to retrieve.

        Returns
        -------
        numpy.ndarray
            1-D array of shape ``(n_wavenumber,)``.

        Raises
        ------
        ValueError
            If the data are 1-D (no scan dimension).
        IndexError
            If ``scan_index`` is outside ``[0, n_scan)``.
        """
        if self.ndim != 2:
            raise ValueError("get_scan requires 2-D data")
        if not (0 <= scan_index < self.shape[0]):
            raise IndexError(
                f"scan_index {scan_index} out of bounds for {self.shape[0]} scans"
            )
        return self.ds["data"].isel(scan=scan_index).values

    def get_baseline_scan(self, scan_index: int) -> npt.NDArray:
        """Return the stored baseline curve of a single scan by integer index.

        Parameters
        ----------
        scan_index : int
            Zero-based index of the scan to retrieve.

        Returns
        -------
        numpy.ndarray
            1-D array of shape ``(n_wavenumber,)``.

        Raises
        ------
        ValueError
            If the data are 1-D (no scan dimension) or if no baseline is stored.
        IndexError
            If ``scan_index`` is outside ``[0, n_scan)``.
        """
        if self.ndim != 2:
            raise ValueError("get_baseline_scan requires 2-D data")
        if not self.has_baseline:
            raise ValueError("No baseline has been stored; run correct_baseline() first.")
        if not (0 <= scan_index < self.shape[0]):
            raise IndexError(
                f"scan_index {scan_index} out of bounds for {self.shape[0]} scans"
            )
        return self.ds["baseline"].isel(scan=scan_index).values

    def get_scan_by_tos(
        self,
        target_tos: Union[float, Sequence[float]],
        method: Literal["nearest", "pad", "backfill"] = "nearest",
        tolerance_seconds: Optional[float] = 10,
    ) -> Union[npt.NDArray]:
        """Return the scan(s) nearest to one or more target elapsed-time values.

        Parameters
        ----------
        target_tos : float or sequence of float
            Target elapsed time(s) in seconds.
        method : {'nearest', 'pad', 'backfill'}, optional
            Interpolation method passed to ``xarray.DataArray.sel``
            (default is ``'nearest'``).
        tolerance_seconds : float or None, optional
            Maximum allowed distance between a target and the nearest scan.
            Raises ``ValueError`` when exceeded.  Pass ``None`` to disable
            (default is ``10``).

        Returns
        -------
        numpy.ndarray
            Shape ``(n_wavenumber,)`` for a scalar target or
            ``(n_targets, n_wavenumber)`` for a sequence.

        Raises
        ------
        ValueError
            If the data are 1-D, if no ``'tos'`` coordinate is present, or if
            any target exceeds ``tolerance_seconds`` from the nearest scan.
        """
        if self.ndim != 2:
            raise ValueError("get_scan_by_tos requires 2-D data")
        if self.tos is None:
            raise ValueError("get_scan_by_tos requires 'tos' coordinate")
        return self._select_var_by_tos(self.ds["data"], target_tos, method, tolerance_seconds)

    def get_baseline_by_tos(
        self,
        target_tos: Union[float, Sequence[float]],
        method: Literal["nearest", "pad", "backfill"] = "nearest",
        tolerance_seconds: Optional[float] = 10,
    ) -> Union[npt.NDArray]:
        """Return the stored baseline curve(s) nearest to one or more target elapsed-time values.

        Parameters
        ----------
        target_tos : float or sequence of float
            Target elapsed time(s) in seconds.
        method : {'nearest', 'pad', 'backfill'}, optional
            Interpolation method passed to ``xarray.DataArray.sel``
            (default is ``'nearest'``).
        tolerance_seconds : float or None, optional
            Maximum allowed distance between a target and the nearest scan.
            Raises ``ValueError`` when exceeded.  Pass ``None`` to disable
            (default is ``10``).

        Returns
        -------
        numpy.ndarray
            Shape ``(n_wavenumber,)`` for a scalar target or
            ``(n_targets, n_wavenumber)`` for a sequence.

        Raises
        ------
        ValueError
            If the data are 1-D, if no baseline is stored, if no ``'tos'``
            coordinate is present, or if any target exceeds
            ``tolerance_seconds`` from the nearest scan.
        """
        if self.ndim != 2:
            raise ValueError("get_baseline_by_tos requires 2-D data")
        if not self.has_baseline:
            raise ValueError("No baseline has been stored; run correct_baseline() first.")
        if self.tos is None:
            raise ValueError("get_baseline_by_tos requires 'tos' coordinate")
        return self._select_var_by_tos(self.ds["baseline"], target_tos, method, tolerance_seconds)

    def get_id_by_tos(
        self,
        target_tos: Union[float, Sequence[float]],
        method: Literal["nearest", "pad", "backfill"] = "nearest",
        tolerance_seconds: Optional[float] = 10,
    ) -> Union[int, npt.NDArray]:
        """Return the scan id(s) nearest to one or more target elapsed-time values.

        Parameters
        ----------
        target_tos : float or sequence of float
            Target elapsed time(s) in seconds.
        method : {'nearest', 'pad', 'backfill'}, optional
            Interpolation method passed to ``xarray.DataArray.sel``
            (default is ``'nearest'``).
        tolerance_seconds : float or None, optional
            Maximum allowed distance between a target and the nearest scan.
            Raises ``ValueError`` when exceeded.  Pass ``None`` to disable
            (default is ``10``).

        Returns
        -------
        int or numpy.ndarray
            Scan id (the ``'scan'`` coordinate value) for a scalar target,
            or an array of scan ids for a sequence of targets.

        Raises
        ------
        ValueError
            If the data are 1-D, if no ``'tos'`` coordinate is present, or if
            any target exceeds ``tolerance_seconds`` from the nearest scan.
        """
        if self.ndim != 2:
            raise ValueError("get_id_by_tos requires 2-D data")
        if self.tos is None:
            raise ValueError("get_id_by_tos requires 'tos' coordinate")

        scalar_input = np.ndim(target_tos) == 0
        targets = [float(target_tos)] if scalar_input else [float(t) for t in target_tos]

        def _id_for(t: float) -> int:
            if tolerance_seconds is not None:
                nearest_dist = float(np.abs(self.tos - t).min())
                if nearest_dist > tolerance_seconds:
                    raise ValueError(
                        f"Requested tos {t:.1f}s is {nearest_dist:.1f}s from the nearest scan "
                        f"(tolerance: {tolerance_seconds:.1f}s)"
                    )
            return int(self.ds["scan"].sel(tos=t, method=method).item())

        ids = np.array([_id_for(t) for t in targets], dtype=int)
        return int(ids[0]) if scalar_input else ids

    def get_scan_by_tos_average(
        self,
        target_tos: Union[float, Sequence[float]],
        method: Literal["nearest", "pad", "backfill"] = "nearest",
        tolerance_seconds: Optional[float] = 10,
        number_of_scans: Optional[int] = None,
        time_window: Optional[float] = None,
        direction: Literal["forward", "backward", "center"] = "center",
    ) -> Union[npt.NDArray]:
        """Return the scan averaged over a window centred on each target tos.

        When neither ``number_of_scans`` nor ``time_window`` is supplied the
        single nearest scan is returned (equivalent to ``get_scan_by_tos``).

        Parameters
        ----------
        target_tos : float or sequence of float
            Target elapsed time(s) in seconds that anchor the averaging window.
        method : {'nearest', 'pad', 'backfill'}, optional
            Selection method for the anchor scan (default is ``'nearest'``).
        tolerance_seconds : float or None, optional
            Maximum distance in seconds between a target and its anchor scan
            (default is ``10``).
        number_of_scans : int or None, optional
            Number of consecutive scans to average.  Mutually exclusive with
            ``time_window``.
        time_window : float or None, optional
            Duration in seconds over which scans are averaged.  Mutually
            exclusive with ``number_of_scans``.
        direction : {'forward', 'backward', 'center'}, optional
            Position of the anchor within the window (default is
            ``'center'``).

        Returns
        -------
        numpy.ndarray
            Shape ``(n_wavenumber,)`` for a scalar target or
            ``(n_targets, n_wavenumber)`` for a sequence.

        Raises
        ------
        ValueError
            If the data are 1-D, if no ``'tos'`` coordinate is present, if
            any target exceeds the tolerance, or if the resulting window is
            empty.
        """
        if self.ndim != 2:
            raise ValueError("get_scan_by_tos_average requires 2-D data")
        if self.tos is None:
            raise ValueError("get_scan_by_tos_average requires 'tos' coordinate")
        if number_of_scans is None and time_window is None:
            return self.get_scan_by_tos(target_tos, method=method, tolerance_seconds=tolerance_seconds)
        if number_of_scans is not None and time_window is not None:
            raise ValueError("Provide exactly one of: number_of_scans or time_window")

        scalar_input = np.ndim(target_tos) == 0
        targets = [float(target_tos)] if scalar_input else [float(t) for t in target_tos]
        tos_values = self.tos  # sorted 1-D array

        def _anchor_index(t: float) -> int:
            """Index of the scan nearest to t, with tolerance check."""
            dists = np.abs(tos_values - t)
            idx = int(dists.argmin())
            if tolerance_seconds is not None and dists[idx] > tolerance_seconds:
                raise ValueError(
                    f"Requested tos {t:.1f}s is {dists[idx]:.1f}s from the nearest scan "
                    f"(tolerance: {tolerance_seconds:.1f}s)"
                )
            return idx

        def _window_indices(anchor_idx: int) -> slice:
            """Return the index slice for the averaging window."""
            n = len(tos_values)

            if number_of_scans is not None:
                half = number_of_scans // 2
                if direction == "center":
                    i0 = anchor_idx - half
                    i1 = anchor_idx + (number_of_scans - half)  # handles odd counts correctly
                elif direction == "forward":
                    i0 = anchor_idx
                    i1 = anchor_idx + number_of_scans
                else:  # backward
                    i0 = anchor_idx - number_of_scans + 1
                    i1 = anchor_idx + 1

            else:  # time_window
                t_anchor = tos_values[anchor_idx]
                half_w = time_window / 2.0
                if direction == "center":
                    t0, t1 = t_anchor - half_w, t_anchor + half_w
                elif direction == "forward":
                    t0, t1 = t_anchor, t_anchor + time_window
                else:  # backward
                    t0, t1 = t_anchor - time_window, t_anchor
                i0 = int(np.searchsorted(tos_values, t0, side="left"))
                i1 = int(np.searchsorted(tos_values, t1, side="right"))

            i0 = max(i0, 0)
            i1 = min(i1, n)

            if i0 >= i1:
                raise ValueError(
                    f"Window [{i0}:{i1}] is empty for anchor index {anchor_idx}. "
                    "Check number_of_scans / time_window against the data range."
                )
            return slice(i0, i1)

        def _average_one(t: float) -> npt.NDArray:
            anchor_idx = _anchor_index(t)
            win = _window_indices(anchor_idx)
            window_data = self.ds["data"].isel(scan=win).values  # shape: (n_scans_in_window, n_masses)
            return window_data.mean(axis=0)

        results = np.vstack([_average_one(t) for t in targets])
        return results[0] if scalar_input else results

    def get_evolution(
        self,
        wavenumber_per_cm: Union[float, list[float], npt.NDArray],
        method: Literal["nearest", "pad", "backfill"] = "nearest",
        tolerance_per_cm: Optional[float] = None,
        rolling_window: Optional[int] = None,
    ) -> xr.DataArray:
        """Return the intensity evolution over scans at one or more wavenumbers.

        Parameters
        ----------
        wavenumber_per_cm : float or array-like
            Target wavenumber(s) in cm⁻¹.
        method : {'nearest', 'pad', 'backfill'}, optional
            Selection method passed to ``xarray.DataArray.sel``
            (default is ``'nearest'``).
        tolerance_per_cm : float or None, optional
            Maximum allowed distance in cm⁻¹ between a target and the nearest
            grid point.  Raises ``ValueError`` when exceeded.  Pass ``None``
            to disable (default is ``None``).
        rolling_window : int or None, optional
            If provided, apply a rolling mean over ``rolling_window`` scans
            (centred, minimum one period) before returning (default is
            ``None``).

        Returns
        -------
        xr.DataArray
            Scalar target in, 1-D out: dims ``('scan',)`` with the selected
            wavenumber kept as a scalar coordinate.  A sequence of targets
            gives dims ``('scan', 'wavenumber')``.

        Raises
        ------
        ValueError
            If the data are 1-D or if a target wavenumber exceeds
            ``tolerance_per_cm`` from the nearest grid point.
        """
        if self.ndim != 2:
            raise ValueError("get_evolution requires 2-D data")

        scalar_input = np.ndim(wavenumber_per_cm) == 0
        targets_si = np.atleast_1d(np.asarray(wavenumber_per_cm, dtype=float)) * 100.0

        if tolerance_per_cm is not None:
            tol_si = tolerance_per_cm * 100.0
            for t in targets_si:
                nearest_dist = float(np.abs(self.wavenumber - t).min())
                if nearest_dist > tol_si:
                    raise ValueError(
                        f"Requested wavenumber {t / 100:.1f} cm⁻¹ is "
                        f"{nearest_dist / 100:.1f} cm⁻¹ from the nearest grid point "
                        f"(tolerance: {tolerance_per_cm:.1f} cm⁻¹)"
                    )

        result = self.ds["data"].sel(wavenumber=targets_si, method=method)

        if rolling_window is not None:
            result = result.rolling(scan=rolling_window, center=True, min_periods=1).mean()

        if scalar_input:
            # Scalar in, scalar out: drop the length-1 wavenumber dim, keeping
            # the selected wavenumber as a scalar coordinate.
            result = result.isel(wavenumber=0)

        return result

    def _select_var_by_tos(
        self,
        var: xr.DataArray,
        target_tos: Union[float, Sequence[float]],
        method: Literal["nearest", "pad", "backfill"],
        tolerance_seconds: Optional[float],
    ) -> npt.NDArray:
        """Select scan(s) of var nearest to one or more target tos values (shared by get_scan_by_tos/get_baseline_by_tos)."""
        scalar_input = np.ndim(target_tos) == 0
        targets = [float(target_tos)] if scalar_input else [float(t) for t in target_tos]

        def _fetch_one(t: float) -> npt.NDArray:
            if tolerance_seconds is not None:
                nearest_dist = float(np.abs(self.tos - t).min())
                if nearest_dist > tolerance_seconds:
                    raise ValueError(
                        f"Requested tos {t:.1f}s is {nearest_dist:.1f}s from the nearest scan "
                        f"(tolerance: {tolerance_seconds:.1f}s)"
                    )
            return var.sel(tos=t, method=method).values

        results = np.vstack([_fetch_one(t) for t in targets])
        return results[0] if scalar_input else results
