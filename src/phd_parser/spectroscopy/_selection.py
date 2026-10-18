"""Cutting a spectrum or a series: by axis window (any unit), by scan index, by tos, by scan id.

A mixin of :class:`~phd_parser.spectroscopy.core.SpectroscopyData`; never used on its own.
"""

import logging
from typing import TYPE_CHECKING, Literal, Optional, Sequence, Tuple
from typing import Literal, Optional, Sequence

import numpy as np
import numpy.typing as npt
import xarray as xr

from phd_parser._selection import EmptyWindow, window_mask
from phd_parser.history import records_step
from phd_parser.spectroscopy.units import quantity_of, to_wavenumber

if TYPE_CHECKING:
    from phd_parser.spectroscopy.core import SpectroscopyData

logger = logging.getLogger(__name__)


class _SelectionMixin:
    """Cutting a spectrum or a series: by axis window (any unit), by scan index, by tos, by scan id."""

    @records_step
    def sort(self, by: str | Sequence[str] = "wavenumber", ascending: bool = True) -> "SpectroscopyData":
        """Return a new instance with coordinates sorted.

        Parameters
        ----------
        by : str or sequence of str, optional
            Coordinate name(s) to sort by (default is ``'wavenumber'``).
        ascending : bool, optional
            Sort in ascending order when ``True`` (default is ``True``).

        Returns
        -------
        SpectroscopyData
            New instance with the dataset sorted along the specified
            coordinate(s).
        """
        ds_sorted = self.ds.sortby(by, ascending=ascending)
        return self._new(ds=ds_sorted)

    @records_step
    def select_by_idx(self, idx: int) -> "SpectroscopyData":
        """Return a 1-D SpectroscopyData containing the scan at the given integer index.

        Parameters
        ----------
        idx : int
            Zero-based scan index.

        Returns
        -------
        SpectroscopyData
            New 1-D instance for the selected scan.

        Raises
        ------
        ValueError
            If the data are already 1-D.
        IndexError
            If ``idx`` is outside ``[0, n_scan)``.
        """
        if self.ndim != 2:
            raise ValueError("select_by_idx requires 2-D data")
        if not (0 <= idx < self.shape[0]):
            raise IndexError(f"idx {idx} out of bounds for {self.shape[0]} scans")
        ds_selected = self.ds.isel(scan=idx)
        return self._new(ds=ds_selected)

    @records_step
    def select_by_tos(self, target_tos: float, method: Literal["nearest", "pad", "backfill"] = "nearest", tolerance_seconds: Optional[float] = 10) -> "SpectroscopyData":
        """Return a 1-D SpectroscopyData containing the scan nearest to ``target_tos``.

        Parameters
        ----------
        target_tos : float
            Target elapsed time in seconds.
        method : {'nearest', 'pad', 'backfill'}, optional
            Selection method (default is ``'nearest'``).
        tolerance_seconds : float or None, optional
            Maximum distance in seconds from the target to the nearest scan.
            Raises ``ValueError`` when exceeded (default is ``10``).

        Returns
        -------
        SpectroscopyData
            New 1-D instance for the selected scan.

        Raises
        ------
        ValueError
            If the data are 1-D, if no ``'tos'`` coordinate is present, or if
            the target exceeds the tolerance.
        """
        if self.ndim != 2:
            raise ValueError("select_by_tos requires 2-D data")
        if self.tos is None:
            raise ValueError("select_by_tos requires 'tos' coordinate")

        if tolerance_seconds is not None:
            nearest_dist = float(np.abs(self.tos - target_tos).min())
            if nearest_dist > tolerance_seconds:
                raise ValueError(
                    f"Requested tos {target_tos:.1f}s is {nearest_dist:.1f}s from the nearest scan "
                    f"(tolerance: {tolerance_seconds:.1f}s)"
                )

        ds_selected = self.ds.sel(tos=target_tos, method=method)
        return self._new(ds=ds_selected)

    @records_step
    def select_wavenumber_range(
        self,
        min_cm: Optional[float] = None,
        max_cm: Optional[float] = None,
    ) -> "SpectroscopyData":
        """Return a new instance restricted to a wavenumber sub-range.

        Parameters
        ----------
        min_cm : float or None, optional
            Lower bound in cm⁻¹ (inclusive).  ``None`` means no lower bound.
        max_cm : float or None, optional
            Upper bound in cm⁻¹ (inclusive).  ``None`` means no upper bound.

        Returns
        -------
        SpectroscopyData
            New instance with the wavenumber axis truncated and the background
            sliced to match.
        """
        da = self.ds["data"]
        if min_cm is not None:
            wn = da.coords["wavenumber"].values
            da = da.sel(wavenumber=wn >= min_cm * 100.0)
        if max_cm is not None:
            wn = da.coords["wavenumber"].values
            da = da.sel(wavenumber=wn <= max_cm * 100.0)

        ds_new = self._build_ds(
            wavenumber_si=da.coords["wavenumber"].values,
            values=da.values,
            tos=da.coords["tos"].values if "tos" in da.coords else None,
            attrs=dict(self.ds.attrs),
        )
        ds_new = self._slice_auxiliary_to(ds_new)
        ds_new = self._slice_baseline_to(ds_new)
        return self._new(ds=ds_new)

    @records_step
    def select_axis_range(
        self,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        unit: str = "cm^-1",
    ) -> "SpectroscopyData":
        """Return a new instance restricted to an axis window given in any spectral unit.

        :meth:`select_wavenumber_range` for a window expressed in ``unit``
        (see :meth:`axis`): ``select_axis_range(2.5, 25, "um")`` keeps the
        mid-infrared, ``select_axis_range(0.1, 0.5, "eV")`` the same in photon
        energy. The bounds are inclusive and either may be ``None``.

        Parameters
        ----------
        minimum, maximum : float or None, optional
            Bounds in ``unit``.
        unit : str, optional
            The unit of the bounds (default is ``'cm^-1'``).

        Returns
        -------
        SpectroscopyData
            New instance with the axis truncated and background, baseline and
            map coordinates sliced to match.
        """
        zero = self._shift_zero_si(unit)
        # widen by a relative 1e-9 so a bound that converts to a grid point keeps that point
        bounds = [to_wavenumber(bound, unit, zero) / 100.0 for bound in (minimum, maximum) if bound is not None]
        bounds = [min(bounds) * (1 - 1e-9), max(bounds) * (1 + 1e-9)] if len(bounds) == 2 else [
            value * (1 - 1e-9) if (minimum is not None) != (quantity_of(unit) == "wavelength") else value * (1 + 1e-9)
            for value in bounds
        ]
        if not bounds:
            return self.select_wavenumber_range(None, None)
        lower = min(bounds) if len(bounds) == 2 else None
        upper = max(bounds) if len(bounds) == 2 else None
        if len(bounds) == 1:
            # one open side: a wavelength bound inverts the direction of the window
            inverted = quantity_of(unit) == "wavelength"
            if (minimum is not None) != inverted:
                lower = bounds[0]
            else:
                upper = bounds[0]
        return self.select_wavenumber_range(lower, upper)

    @records_step
    def select_shift_range(
        self,
        min_shift_per_cm: Optional[float] = None,
        max_shift_per_cm: Optional[float] = None,
    ) -> "SpectroscopyData":
        """Return a new instance restricted to a Raman shift range in cm⁻¹ (inclusive).

        :meth:`select_wavenumber_range` under its Raman name.

        Parameters
        ----------
        min_shift_per_cm : float or None, optional
            Lower bound in cm⁻¹ (default is ``None``, no lower bound).
        max_shift_per_cm : float or None, optional
            Upper bound in cm⁻¹ (default is ``None``, no upper bound).

        Returns
        -------
        SpectroscopyData
            New instance with the shift axis truncated.
        """
        return self.select_wavenumber_range(min_shift_per_cm, max_shift_per_cm)

    @records_step
    def select_wavenumber_index_range(
        self,
        min_idx: Optional[int] = None,
        max_idx: Optional[int] = None,
    ) -> "SpectroscopyData":
        """Return a new instance restricted to a wavenumber sub-range selected by position.

        Unlike ``select_wavenumber_range``, bounds are given as zero-based
        positional indices into the wavenumber axis rather than cm⁻¹ values.

        Parameters
        ----------
        min_idx : int or None, optional
            Lower bound index (inclusive).  ``None`` means index ``0``.
        max_idx : int or None, optional
            Upper bound index (inclusive).  ``None`` means the last index.

        Returns
        -------
        SpectroscopyData
            New instance with the wavenumber axis truncated and the background
            sliced to match.

        Raises
        ------
        IndexError
            If ``min_idx`` or ``max_idx`` is outside ``[0, n_wavenumber)``.
        """
        n = self.shape[-1]
        lo = 0 if min_idx is None else min_idx
        hi = n - 1 if max_idx is None else max_idx
        if not (0 <= lo < n) or not (0 <= hi < n):
            raise IndexError(f"index range [{lo}, {hi}] out of bounds for {n} wavenumber points")

        da = self.ds["data"].isel(wavenumber=slice(lo, hi + 1))
        ds_new = self._build_ds(
            wavenumber_si=da.coords["wavenumber"].values,
            values=da.values,
            tos=da.coords["tos"].values if "tos" in da.coords else None,
            attrs=dict(self.ds.attrs),
        )
        ds_new = self._slice_auxiliary_to(ds_new)
        ds_new = self._slice_baseline_to(ds_new)
        return self._new(ds=ds_new)

    @records_step
    def select_tos_range(
        self,
        min_s: Optional[float] = None,
        max_s: Optional[float] = None,
        empty: EmptyWindow = "warn",
    ) -> "SpectroscopyData":
        """Return a new instance restricted to scans within a tos sub-range.

        Parameters
        ----------
        min_s : float or None, optional
            Lower bound in seconds (inclusive).  ``None`` means no lower bound.
        max_s : float or None, optional
            Upper bound in seconds (inclusive).  ``None`` means no upper bound.
        empty : {"warn", "raise"}, optional
            What to do when nothing falls inside the window: log it and
            return the empty selection (default), or raise ``ValueError``.

        Returns
        -------
        SpectroscopyData
            New instance containing only the scans within the specified range.

        Raises
        ------
        ValueError
            If no ``'tos'`` coordinate is present, or if the window is empty
            and ``empty="raise"``.
        """
        if self.tos is None:
            raise ValueError("select_tos_range requires a 'tos' coordinate")

        # Operate on the whole Dataset so 'background' and 'baseline' (if present) are
        # carried through the scan-axis selection automatically.
        mask = window_mask(self.ds.coords["tos"].values, min_s, max_s,
                           what="scan", axis="tos", unit="s", empty=empty)
        # tos values are absolute elapsed seconds, so tos_start + tos[i] remains valid
        return self._new(ds=self.ds.isel(scan=mask))

    @records_step
    def select_scan_id_range(
        self,
        min_id: Optional[int] = None,
        max_id: Optional[int] = None,
        empty: EmptyWindow = "warn",
    ) -> "SpectroscopyData":
        """Return a new instance restricted to scans within a scan-id sub-range.

        Unlike ``select_tos_range``, bounds are given as scan ids (the
        ``'scan'`` coordinate, see ``get_id_by_tos``) rather than elapsed time.

        Parameters
        ----------
        min_id : int or None, optional
            Lower bound on the scan id (inclusive).  ``None`` means no lower bound.
        max_id : int or None, optional
            Upper bound on the scan id (inclusive).  ``None`` means no upper bound.
        empty : {"warn", "raise"}, optional
            What to do when nothing falls inside the window: log it and
            return the empty selection (default), or raise ``ValueError``.

        Returns
        -------
        SpectroscopyData
            New instance containing only the scans within the specified id range.

        Raises
        ------
        ValueError
            If the data are 1-D (no scan dimension), or if the range is empty
            and ``empty="raise"``.
        """
        if self.ndim != 2:
            raise ValueError("select_scan_id_range requires 2-D data")

        mask = window_mask(self.ds.coords["scan"].values, min_id, max_id,
                           what="scan", axis="scan id", unit="", empty=empty)
        return self._new(ds=self.ds.isel(scan=mask))

    @records_step
    def mask_wavenumber_ranges(self, ranges_cm: Sequence[Tuple[float, float]]) -> "SpectroscopyData":
        """Set the data in the given wavenumber ranges to NaN, keeping the axis intact.

        Use for regions that cannot be corrected, e.g. a saturated gas-phase
        band; the result still lines up with unmasked data.

        Parameters
        ----------
        ranges_cm : sequence of (float, float)
            ``(low, high)`` wavenumber bounds in cm⁻¹, inclusive; the order
            within a pair does not matter.

        Returns
        -------
        SpectroscopyData
            New instance with NaN in the masked ranges. The ranges are
            recorded flat, ``[low1, high1, low2, high2, ...]``, in
            ``ds.attrs['masked_ranges_cm']`` (NetCDF cannot store nested lists).
        """
        wavenumber_cm = self.wavenumber_per_cm
        mask = np.zeros(wavenumber_cm.size, dtype=bool)
        for low, high in ranges_cm:
            mask |= (wavenumber_cm >= min(low, high)) & (wavenumber_cm <= max(low, high))
        mask_da = xr.DataArray(mask, coords={"wavenumber": self.ds.coords["wavenumber"]}, dims=["wavenumber"])
        ds_new = self.ds.assign(data=self.ds["data"].where(~mask_da))
        previous = list(self.ds.attrs.get("masked_ranges_cm", []))
        ds_new.attrs = {**self.ds.attrs,
                        "masked_ranges_cm": previous + [float(value) for pair in ranges_cm for value in sorted(pair)]}
        logger.debug(f"Masked {mask.sum()} of {mask.size} wavenumber points")
        return self._new(ds=ds_new)

    @records_step
    def drop_scans(self, drop: npt.NDArray[np.bool_]) -> "SpectroscopyData":
        """Remove scans by boolean mask, keeping ``tos``, background and baseline aligned.

        Parameters
        ----------
        drop : numpy.ndarray of bool
            Shape ``(n_scan,)``, ``True`` for the scans to remove. A mask
            longer than 20 scans is kept in the history only as a summary;
            :meth:`drop_spike_scans` is the replayable form.

        Returns
        -------
        SpectroscopyData
            New instance without the dropped scans. The number dropped is
            added to ``ds.attrs['n_scans_dropped']``.

        Raises
        ------
        ValueError
            If the data are 1-D or ``drop`` has the wrong shape.
        """
        drop = np.asarray(drop, dtype=bool)
        if self.ndim != 2:
            raise ValueError("drop_scans requires 2-D data")
        if drop.shape != (self.shape[0],):
            raise ValueError(f"drop has shape {drop.shape}, expected ({self.shape[0]},)")
        ds_new = self.ds.isel(scan=~drop)
        ds_new.attrs = {**self.ds.attrs,
                        "n_scans_dropped": int(self.ds.attrs.get("n_scans_dropped", 0)) + int(drop.sum())}
        return self._new(ds=ds_new)

    @records_step
    def drop_spike_scans(self, window: int = 7, threshold: float = 6.0, min_fraction: float = 0.1) -> "SpectroscopyData":
        """Drop the scans :meth:`find_spike_scans` flags, in one recorded, replayable step.

        Parameters
        ----------
        window, threshold, min_fraction
            As in :meth:`find_spike_scans`.

        Returns
        -------
        SpectroscopyData
            New instance without the spike scans (``self`` if there are none).
        """
        spikes = self.find_spike_scans(window=window, threshold=threshold, min_fraction=min_fraction)
        if not spikes.any():
            return self
        return self.drop_scans(spikes)
