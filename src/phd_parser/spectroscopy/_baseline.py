"""Baseline correction (offset, PCHIP) and the stored baseline curve that makes it reversible.

A mixin of :class:`~phd_parser.spectroscopy.core.SpectroscopyData`; never used on its own.
"""

import logging
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import xarray as xr

from phd_parser.history import records_step

if TYPE_CHECKING:
    from phd_parser.spectroscopy.core import SpectroscopyData

logger = logging.getLogger(__name__)


def full_cycle_starts(tos: npt.NDArray, cycle_starts: Sequence[float], period: float) -> npt.NDArray:
    """The cycle starts whose whole period is covered by ``tos``.

    Parameters
    ----------
    tos : numpy.ndarray
        Elapsed seconds of the scans, ascending.
    cycle_starts : sequence of float
        Start of each period in the same time base.
    period : float
        Length of a period in seconds.

    Returns
    -------
    numpy.ndarray
        The starts with ``start >= tos[0]`` and ``start + period <= tos[-1]``.
    """
    starts = np.asarray(cycle_starts, dtype=float)
    return starts[(starts >= tos[0]) & (starts + period <= tos[-1])]


class _BaselineMixin:
    """Baseline correction (offset, PCHIP) and the stored baseline curve that makes it reversible."""

    @records_step
    def del_baseline(self) -> "SpectroscopyData":
        """Remove the stored baseline curve without changing data values.

        Returns
        -------
        SpectroscopyData
            New instance with the ``'baseline'`` variable dropped.
        """
        return self._del_baseline()

    @records_step
    def unbaseline(self) -> "SpectroscopyData":
        """Return a new instance with the baseline added back into ``data`` and dropped.

        Lets the regular getters (``get_scan``, ``get_scan_by_tos``,
        ``get_evolution``, ...) return pre-correction values, e.g.
        ``ir.unbaseline().get_scan_by_tos(120)``.

        Returns
        -------
        SpectroscopyData
            New instance with ``data`` equal to ``data_unbaselined`` and no
            stored baseline.  Returns ``self`` unchanged if no baseline is
            stored.
        """
        if not self.has_baseline:
            return self
        ds_new = self._build_ds(
            self.wavenumber, self.data_unbaselined, tos=self.tos, attrs=dict(self.ds.attrs)
        )
        return self._new(ds=self._carry_auxiliary(ds_new))

    @records_step
    def correct_offset(
        self,
        anchor_range_cm: Tuple[float, float] = (2500, 2600),
    ) -> "SpectroscopyData":
        """Subtract a constant offset computed as the mean over an anchor range.

        Parameters
        ----------
        anchor_range_cm : tuple of float, optional
            ``(low, high)`` wavenumber bounds in cm⁻¹ that define the
            baseline anchor region (default is ``(2500, 2600)``).

        Returns
        -------
        SpectroscopyData
            New instance with the offset removed.  The subtracted offset is
            added to (or stored as) ``ds['baseline']``; recover the
            pre-correction values via ``data_unbaselined``.

        Raises
        ------
        ValueError
            If no wavenumber points fall within the anchor range.
        """
        lo_si = min(anchor_range_cm) * 100.0
        hi_si = max(anchor_range_cm) * 100.0
        wn = self.wavenumber
        mask = (wn >= lo_si) & (wn <= hi_si)
        if not mask.any():
            raise ValueError(
                f"No wavenumber points in anchor range "
                f"{min(anchor_range_cm):.0f}–{max(anchor_range_cm):.0f} cm⁻¹"
            )

        if self.ndim == 1:
            offset = self.values[mask].mean()
            offset_arr = np.full_like(self.values, offset)
        else:
            offset = self.values[..., mask].mean(axis=-1, keepdims=True)
            offset_arr = np.broadcast_to(offset, self.values.shape).copy()
        corrected = self.values - offset_arr

        new_baseline = self._accumulate_baseline(offset_arr)
        new_attrs = {**self.ds.attrs, "baseline_anchor_range_cm": list(anchor_range_cm)}
        ds_new = self._build_ds(wn, corrected, tos=self.tos, attrs=new_attrs)
        ds_new = self._with_baseline(ds_new, new_baseline)
        return self._new(ds=self._carry_auxiliary(ds_new))

    @records_step
    def correct_pchip(
        self,
        control_points_cm: Sequence[float],
        point_avg_half_width: int = 0,
    ) -> "SpectroscopyData":
        """Subtract a PCHIP-interpolated baseline through the given control points.

        Parameters
        ----------
        control_points_cm : sequence of float
            Wavenumber positions in cm⁻¹ used as knots for the PCHIP spline.
        point_avg_half_width : int, optional
            Half-width (in points) of the local averaging window used to
            compute the intensity at each knot (default is ``0``, i.e. the
            single nearest point).

        Returns
        -------
        SpectroscopyData
            New instance with the PCHIP baseline subtracted.  The subtracted
            curve is added to (or stored as) ``ds['baseline']``; recover the
            pre-correction values via ``data_unbaselined``.
        """
        from scipy.interpolate import PchipInterpolator

        wn_si = self.wavenumber
        wn_cm = wn_si / 100.0
        control_points = np.sort(np.asarray(control_points_cm, dtype=float))

        # The knots are the same for every scan: locate them once, then
        # interpolate all scans in one call.
        knot_index = np.array([int(np.abs(wn_cm - point).argmin()) for point in control_points])
        x_knots = wn_cm[knot_index]
        spectra = self.values.reshape(-1, wn_cm.size)
        y_knots = np.stack(
            [
                spectra[:, max(0, idx - point_avg_half_width): idx + point_avg_half_width + 1].mean(axis=1)
                for idx in knot_index
            ],
            axis=1,
        )
        pchip_curve = PchipInterpolator(x_knots, y_knots, axis=1)(wn_cm).reshape(self.values.shape)
        corrected = self.values - pchip_curve

        new_baseline = self._accumulate_baseline(pchip_curve)
        new_attrs = {
            **self.ds.attrs,
            "baseline_pchip_control_points_cm": sorted(control_points_cm),
            "baseline_pchip_half_width": point_avg_half_width,
        }
        ds_new = self._build_ds(wn_si, corrected, tos=self.tos, attrs=new_attrs)
        ds_new = self._with_baseline(ds_new, new_baseline)
        return self._new(ds=self._carry_auxiliary(ds_new))

    @records_step
    def correct_baseline(
        self,
        anchor_range_cm: Tuple[float, float] = (2500, 2600),
        control_points_cm: Optional[Sequence[float]] = None,
        point_avg_half_width: int = 0,
        double_offset: bool = True,
    ) -> "SpectroscopyData":
        """Apply a two-step baseline correction: offset then optional PCHIP.

        Step 1 subtracts a constant offset via ``correct_offset``.  If
        ``control_points_cm`` is provided, step 2 removes a PCHIP spline via
        ``correct_pchip``, followed optionally by a second offset step (mirrors
        DRIFTS behaviour).

        Parameters
        ----------
        anchor_range_cm : tuple of float, optional
            Anchor wavenumber range in cm⁻¹ used for the offset step(s)
            (default is ``(2500, 2600)``).
        control_points_cm : sequence of float or None, optional
            Knot positions in cm⁻¹ for the PCHIP step.  ``None`` skips the
            PCHIP step (default is ``None``).
        point_avg_half_width : int, optional
            Half-width in points for local averaging at each PCHIP knot
            (default is ``0``).
        double_offset : bool, optional
            Apply a second offset correction after the PCHIP step when
            ``True`` (default is ``True``).

        Returns
        -------
        SpectroscopyData
            New instance with the baseline removed.  The cumulative baseline
            curve is stored in ``ds['baseline']``; recover the
            pre-correction values via ``data_unbaselined``.
        """
        # Step 1: offset, step 2: PCHIP, step 3: optional second offset (mirrors DRIFTS behaviour)
        result = self.correct_offset(anchor_range_cm)
        if control_points_cm:
            result = result.correct_pchip(control_points_cm, point_avg_half_width)
            if double_offset:
                result = result.correct_offset(anchor_range_cm)
        return result

    @records_step
    def reapply_baseline(self) -> "SpectroscopyData":
        """Re-run baseline correction using parameters stored in dataset attributes.

        Returns
        -------
        SpectroscopyData
            New instance with the stored baseline correction reapplied.

        Raises
        ------
        ValueError
            If ``ds.attrs`` does not contain ``'baseline_anchor_range_cm'``.
        """
        # Re-runs correction using parameters stored in attributes (e.g. after average_scans)
        anchor_range_cm = self.ds.attrs.get("baseline_anchor_range_cm")
        if anchor_range_cm is None:
            raise ValueError("No baseline parameters found in attributes.")

        # Restore the unbaselined values first so correct_baseline() isn't applied on top
        # of the previously stored baseline (which would double-correct).
        base = self
        if self.has_baseline:
            restored_ds = self._build_ds(
                self.wavenumber, self.data_unbaselined, tos=self.tos, attrs=dict(self.ds.attrs)
            )
            base = self._new(ds=self._carry_auxiliary(restored_ds))

        return base.correct_baseline(
            anchor_range_cm=tuple(anchor_range_cm),
            control_points_cm=self.ds.attrs.get("baseline_pchip_control_points_cm"),
            point_avg_half_width=self.ds.attrs.get("baseline_pchip_half_width", 0),
        )

    def _del_baseline(self) -> "SpectroscopyData":
        if "baseline" not in self.ds:
            return self
        return self._new(ds=self.ds.drop_vars("baseline"))

    def _accumulate_baseline(self, increment: npt.NDArray) -> npt.NDArray:
        """Add increment to the existing stored baseline, or return it as the new baseline."""
        if self.has_baseline:
            return self.baseline + increment
        return increment

    def _with_baseline(self, ds: xr.Dataset, baseline_values: Optional[npt.NDArray]) -> xr.Dataset:
        """Attach baseline_values to ds, shaped/coordinated like ds['data']."""
        if baseline_values is None:
            return ds
        da = xr.DataArray(
            data=baseline_values, coords=ds["data"].coords, dims=ds["data"].dims, name="baseline"
        )
        return ds.assign({"baseline": da})

    def _carry_baseline(self, ds: xr.Dataset) -> xr.Dataset:
        """Copy baseline variable from self into ds unchanged (dims/shape must match)."""
        if "baseline" in self.ds:
            return ds.assign({"baseline": self.ds["baseline"]})
        return ds

    def _slice_baseline_to(self, ds: xr.Dataset) -> xr.Dataset:
        """Slice baseline to match ds's wavenumber coordinate, then copy into ds."""
        if "baseline" not in self.ds:
            return ds
        bl_sliced = self.ds["baseline"].sel(wavenumber=ds.coords["wavenumber"])
        return ds.assign({"baseline": bl_sliced})

    def _warn_drop_baseline(self, reason: str) -> None:
        if self.has_baseline:
            logger.warning(
                f"Dropping stored baseline: {reason}. Re-run baseline correction afterward if needed."
            )

    @records_step
    def correct_drift(self, cycle_starts: Sequence[float], period: float, order: int = 1) -> "SpectroscopyData":
        """Subtract a slow polynomial drift in time, fitted to the per-cycle means of a modulation experiment.

        Fitting the trend to one mean per full cycle, instead of to every scan,
        keeps the periodic response out of the fit: a line fitted directly to a
        periodic signal picks up part of it as slope.

        Parameters
        ----------
        cycle_starts : sequence of float
            Start of each period in seconds (same time base as ``tos``). Only
            cycles fully inside the data are used.
        period : float
            Modulation period ``T`` in seconds.
        order : int, optional
            Polynomial order of the drift in time (default is ``1``, linear).

        Returns
        -------
        SpectroscopyData
            New instance with the drift subtracted from every scan. The drift is
            added to ``ds['baseline']``; recover the uncorrected values via
            ``data_unbaselined``.

        Raises
        ------
        ValueError
            If the data are 1-D, or fewer than ``order + 1`` full cycles are
            available.
        """
        if self.ndim != 2 or self.tos is None:
            raise ValueError("correct_drift requires 2-D data with a 'tos' coordinate")
        tos = self.tos
        starts = full_cycle_starts(tos, cycle_starts, period)
        if starts.size < order + 1:
            raise ValueError(f"need at least {order + 1} full cycles for order {order}, got {starts.size}")

        means = np.stack([self.values[(tos >= start) & (tos < start + period)].mean(axis=0) for start in starts])
        centres = starts + period / 2
        # Centre time for conditioning; polyfit fits every wavenumber column at once.
        t_ref = centres.mean()
        finite = np.isfinite(means).all(axis=0)
        coefficients = np.full((order + 1, means.shape[1]), np.nan)
        coefficients[:, finite] = np.polyfit(centres - t_ref, means[:, finite], deg=order)
        # Zero the constant term: remove the change over time, keep the absolute level.
        coefficients[-1] = 0.0
        powers = (tos - t_ref)[:, None] ** np.arange(order, -1, -1)[None, :]
        drift = powers @ np.nan_to_num(coefficients)

        ds_new = self.ds.assign(data=self.ds["data"] - drift)
        drift_baseline = self.ds["baseline"].values + drift if "baseline" in self.ds else drift
        ds_new = ds_new.assign(baseline=(self.ds["data"].dims, drift_baseline))
        ds_new.attrs = {**self.ds.attrs, "drift_order": order, "drift_n_cycles": int(starts.size)}
        return self._new(ds=ds_new)
