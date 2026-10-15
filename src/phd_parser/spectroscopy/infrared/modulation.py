"""Modulation excitation spectroscopy (MES) helpers for :class:`IRData`.

Preprocessing and cycle folding for periodic-stimulation experiments (e.g.
switching the feed between H2 and CO2 + H2). Every function takes an
``IRData`` and returns a new object; nothing is modified in place, in keeping
with the ``IRData`` convention.

Cycle times are plain ``tos`` values in seconds (the start of each period,
usually the moment the stimulus is switched on), so this module does not
depend on how the switches were logged.
"""

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import xarray as xr

from phd_parser.spectroscopy.infrared.core import IRData

logger = logging.getLogger(__name__)


def mask_wavenumber_ranges(ir: IRData, ranges_cm: Sequence[Tuple[float, float]]) -> IRData:
    """Set the data in the given wavenumber ranges to NaN.

    Use for regions that cannot be corrected, e.g. a saturated gas-phase
    band. The wavenumber axis is kept intact so the result still lines up
    with unmasked data.

    Parameters
    ----------
    ir : IRData
        Input data, 1-D or 2-D.
    ranges_cm : sequence of (float, float)
        ``(low, high)`` wavenumber bounds in cm⁻¹, inclusive, order within a
        pair does not matter.

    Returns
    -------
    IRData
        New instance with NaN in the masked ranges. The ranges are recorded
        flat, ``[low1, high1, low2, high2, ...]``, in
        ``ds.attrs['masked_ranges_cm']`` (NetCDF cannot store nested lists).
    """
    wn_cm = ir.wavenumber_per_cm
    mask = np.zeros(wn_cm.size, dtype=bool)
    for lo, hi in ranges_cm:
        mask |= (wn_cm >= min(lo, hi)) & (wn_cm <= max(lo, hi))
    mask_da = xr.DataArray(mask, coords={"wavenumber": ir.ds.coords["wavenumber"]}, dims=["wavenumber"])
    ds_new = ir.ds.assign(data=ir.ds["data"].where(~mask_da))
    previous = list(ir.ds.attrs.get("masked_ranges_cm", []))
    ds_new.attrs = {**ir.ds.attrs, "masked_ranges_cm": previous + [float(v) for r in ranges_cm for v in sorted(r)]}
    logger.debug(f"Masked {mask.sum()} of {mask.size} wavenumber points")
    return IRData(ds=ds_new)


def find_spike_scans(
    ir: IRData,
    window: int = 7,
    threshold: float = 6.0,
    min_fraction: float = 0.1,
) -> npt.NDArray[np.bool_]:
    """Flag scans that jump away from their neighbours in time (Hampel test).

    Every point is compared with the running median over ``window`` scans at
    its wavenumber, and flagged when it deviates by more than ``threshold``
    noise standard deviations. A scan is a spike when at least
    ``min_fraction`` of its (finite) wavenumbers are flagged.

    The noise of each wavenumber is estimated from the second difference in
    time, ``1.4826 * MAD(x[i+1] - 2 x[i] + x[i-1]) / sqrt(6)``, which a slow
    or linear change does not bias, so genuine transients such as the
    response right after a feed switch are not mistaken for spikes.

    Parameters
    ----------
    ir : IRData
        2-D input data.
    window : int, optional
        Length of the running window in scans, odd (default is ``7``).
    threshold : float, optional
        Per-point threshold in local robust standard deviations (default is
        ``6.0``).
    min_fraction : float, optional
        Fraction of wavenumbers that must be flagged for the whole scan to
        count as a spike (default is ``0.1``).

    Returns
    -------
    numpy.ndarray of bool
        Shape ``(n_scan,)``, ``True`` for a spike scan.

    Raises
    ------
    ValueError
        If the data are 1-D or ``window`` is not an odd integer >= 3.
    """
    from scipy.ndimage import median_filter

    if ir.ndim != 2:
        raise ValueError("find_spike_scans requires 2-D data")
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3")
    values = ir.values[:, np.isfinite(ir.values).all(axis=0)]
    running_median = median_filter(values, size=(window, 1), mode="nearest")
    deviation = np.abs(values - running_median)
    second_difference = np.diff(values, n=2, axis=0)
    noise = 1.4826 * np.median(np.abs(second_difference), axis=0) / np.sqrt(6.0)
    flagged = deviation > threshold * np.where(noise > 0, noise, np.inf)
    return flagged.mean(axis=1) >= min_fraction


def drop_scans(ir: IRData, drop: npt.NDArray[np.bool_]) -> IRData:
    """Remove scans by boolean mask, keeping ``tos``, background and baseline aligned.

    Parameters
    ----------
    ir : IRData
        2-D input data.
    drop : numpy.ndarray of bool
        Shape ``(n_scan,)``, ``True`` for scans to remove.

    Returns
    -------
    IRData
        New instance without the dropped scans. The number dropped is added
        to ``ds.attrs['n_scans_dropped']``.

    Raises
    ------
    ValueError
        If the data are 1-D or ``drop`` has the wrong shape.
    """
    drop = np.asarray(drop, dtype=bool)
    if ir.ndim != 2:
        raise ValueError("drop_scans requires 2-D data")
    if drop.shape != (ir.shape[0],):
        raise ValueError(f"drop has shape {drop.shape}, expected ({ir.shape[0]},)")
    ds_new = ir.ds.isel(scan=~drop)
    ds_new.attrs = {**ir.ds.attrs, "n_scans_dropped": int(ir.ds.attrs.get("n_scans_dropped", 0)) + int(drop.sum())}
    return IRData(ds=ds_new)


def _full_cycle_starts(tos: npt.NDArray, cycle_starts: Sequence[float], period: float) -> npt.NDArray:
    """Cycle starts whose whole period is covered by ``tos``."""
    starts = np.asarray(cycle_starts, dtype=float)
    return starts[(starts >= tos[0]) & (starts + period <= tos[-1])]


def correct_drift(
    ir: IRData,
    cycle_starts: Sequence[float],
    period: float,
    order: int = 1,
) -> IRData:
    """Subtract a slow polynomial drift in time, fitted to the per-cycle means.

    Fitting the trend to one mean per full cycle, instead of to every scan,
    keeps the periodic response out of the fit: a line fitted directly to a
    periodic signal picks up part of it as slope.

    Parameters
    ----------
    ir : IRData
        2-D input data with a ``tos`` coordinate.
    cycle_starts : sequence of float
        Start of each period in seconds (same time base as ``ir.tos``). Only
        cycles fully inside the data are used.
    period : float
        Modulation period ``T`` in seconds.
    order : int, optional
        Polynomial order of the drift in time (default is ``1``, linear).

    Returns
    -------
    IRData
        New instance with the drift subtracted from every scan. The drift is
        added to ``ds['baseline']``; recover the uncorrected values via
        ``data_unbaselined``.

    Raises
    ------
    ValueError
        If the data are 1-D, or fewer than ``order + 1`` full cycles are
        available.
    """
    if ir.ndim != 2 or ir.tos is None:
        raise ValueError("correct_drift requires 2-D data with a 'tos' coordinate")
    tos = ir.tos
    starts = _full_cycle_starts(tos, cycle_starts, period)
    if starts.size < order + 1:
        raise ValueError(f"need at least {order + 1} full cycles for order {order}, got {starts.size}")

    means = np.stack([ir.values[(tos >= s) & (tos < s + period)].mean(axis=0) for s in starts])
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

    ds_new = ir.ds.assign(data=ir.ds["data"] - drift)
    baseline = ir.ds["baseline"].values + drift if "baseline" in ir.ds else drift
    ds_new = ds_new.assign(baseline=(ir.ds["data"].dims, baseline))
    ds_new.attrs = {**ir.ds.attrs, "drift_order": order, "drift_n_cycles": int(starts.size)}
    return IRData(ds=ds_new)


def fold_cycles(
    ir: IRData,
    cycle_starts: Sequence[float],
    period: float,
    n_discard: int = 2,
    time_step: Optional[float] = None,
) -> xr.DataArray:
    """Cut the time series into periods and put them on a common time grid.

    Parameters
    ----------
    ir : IRData
        2-D input data with a ``tos`` coordinate.
    cycle_starts : sequence of float
        Start of each period in seconds, e.g. the logged switch-on times.
        Taking the logged time of every switch, rather than ``k * period``,
        keeps small timing jitter from smearing the average.
    period : float
        Modulation period ``T`` in seconds.
    n_discard : int, optional
        Number of leading full cycles to throw away while the surface settles
        into its periodic steady state (default is ``2``).
    time_step : float or None, optional
        Spacing of the common time grid in seconds. ``None`` uses the median
        scan spacing.

    Returns
    -------
    xarray.DataArray
        Dims ``('cycle', 'time', 'wavenumber')``: ``cycle`` holds the start
        ``tos`` of each kept cycle, ``time`` runs over ``[0, period)`` in s,
        ``wavenumber`` is in m⁻¹ as in ``IRData``. Attrs carry ``period`` and
        ``n_discarded``.

    Raises
    ------
    ValueError
        If the data are 1-D, or no full cycle is left after discarding.
    """
    from scipy.interpolate import interp1d

    if ir.ndim != 2 or ir.tos is None:
        raise ValueError("fold_cycles requires 2-D data with a 'tos' coordinate")
    tos = ir.tos
    starts = _full_cycle_starts(tos, cycle_starts, period)
    n_incomplete = len(cycle_starts) - starts.size
    if n_incomplete:
        logger.info(f"{n_incomplete} cycle(s) not fully covered by the data were skipped")
    starts = starts[n_discard:]
    if starts.size == 0:
        raise ValueError(f"no full cycle left after discarding {n_discard}")

    if time_step is None:
        time_step = float(np.median(np.diff(tos)))
    time = np.arange(0.0, period, time_step)
    interpolate = interp1d(tos, ir.values, axis=0, assume_sorted=True)
    folded = np.stack([interpolate(s + time) for s in starts])

    return xr.DataArray(
        folded,
        dims=("cycle", "time", "wavenumber"),
        coords={"cycle": starts, "time": time, "wavenumber": ir.ds.coords["wavenumber"].values},
        name="data",
        attrs={
            "period": float(period),
            "n_discarded": int(n_discard),
            "data_type": ir.ds.attrs.get("data_type", ""),
        },
    )


def average_cycles(folded: xr.DataArray) -> IRData:
    """Average folded cycles into one period.

    Parameters
    ----------
    folded : xarray.DataArray
        Output of :func:`fold_cycles`.

    Returns
    -------
    IRData
        One period, ``tos`` is the time within the period in seconds and no
        ``tos_start`` is set. ``ds.attrs`` carries ``modulation_period`` and
        ``n_cycles_averaged``.
    """
    mean = folded.mean(dim="cycle").values
    ir = IRData.from_arrays(
        folded.coords["wavenumber"].values / 100.0,
        mean,
        tos=folded.coords["time"].values,
        data_type=folded.attrs.get("data_type") or "absorbance",
    )
    ds_new = ir.ds.copy()
    ds_new.attrs = {
        **ir.ds.attrs,
        "modulation_period": folded.attrs["period"],
        "n_cycles_averaged": int(folded.sizes["cycle"]),
        "n_cycles_discarded": folded.attrs.get("n_discarded", 0),
    }
    return IRData(ds=ds_new)


def _quadrature_components(
    ir_period: IRData,
    harmonics: Sequence[int],
    period: Optional[float],
) -> Tuple[npt.NDArray, npt.NDArray, float]:
    """In-phase and quadrature parts ``(2/T) int A sin(k w t) dt`` and ``(2/T) int A cos(k w t) dt``."""
    if ir_period.ndim != 2 or ir_period.tos is None:
        raise ValueError("demodulation requires one averaged period as 2-D data with a 'tos' coordinate")
    if period is None:
        period = ir_period.ds.attrs.get("modulation_period")
        if period is None:
            raise ValueError("period not given and 'modulation_period' not in ds.attrs")
    time = ir_period.tos
    step = np.diff(time)
    uniform = np.allclose(step, step[0], rtol=1e-6)
    covers_period = np.isclose(time[0], 0.0, atol=1e-9 * period) and np.isclose(time[-1] + step[0], period, rtol=1e-6)
    if not (uniform and covers_period):
        raise ValueError("time must be a uniform grid covering [0, period), as returned by average_cycles")
    omega = 2.0 * np.pi / period
    k = np.asarray(harmonics, dtype=float)[:, None]
    # Rectangle rule on a uniform periodic grid: exact for harmonics below the Nyquist limit.
    weight = 2.0 * step[0] / period
    in_phase = weight * np.sin(k * omega * time[None, :]) @ ir_period.values
    quadrature = weight * np.cos(k * omega * time[None, :]) @ ir_period.values
    return in_phase, quadrature, float(period)


def demodulate(
    ir_period: IRData,
    harmonics: Sequence[int] = (1, 2, 3),
    phases_deg: Optional[Sequence[float]] = None,
    period: Optional[float] = None,
) -> xr.DataArray:
    """Phase-sensitive detection of one averaged modulation period.

    Computes

        A_k(nu, phi) = (2/T) int_0^T A(nu, t) sin(k w t + phi) dt,   w = 2 pi / T

    for every harmonic ``k`` and demodulation phase angle ``phi``.

    Parameters
    ----------
    ir_period : IRData
        One averaged period, as returned by :func:`average_cycles`: ``tos``
        is the time within the period on a uniform grid over ``[0, T)``.
    harmonics : sequence of int, optional
        Harmonics ``k`` to demodulate at (default is ``(1, 2, 3)``).
    phases_deg : sequence of float or None, optional
        Demodulation phase angles in degrees. ``None`` uses 0 to 350 in steps
        of 10.
    period : float or None, optional
        Modulation period ``T`` in seconds. ``None`` reads
        ``ds.attrs['modulation_period']``.

    Returns
    -------
    xarray.DataArray
        Dims ``('harmonic', 'phase', 'wavenumber')``, ``phase`` in degrees,
        ``wavenumber`` in m⁻¹ as in ``IRData``. A NaN (masked) wavenumber
        stays NaN.

    Raises
    ------
    ValueError
        If the input is not one period on a uniform grid, or no period is
        available.

    Notes
    -----
    The integral is linear in ``sin(phi)`` and ``cos(phi)``, so it is
    evaluated once as ``A_k(phi) = I_k cos(phi) + Q_k sin(phi)`` with the
    in-phase part ``I_k = (2/T) int A sin(k w t) dt`` and the quadrature part
    ``Q_k = (2/T) int A cos(k w t) dt``.

    With this sign convention a response ``a sin(k w t - delta)``, lagging
    the stimulus by ``delta``, gives ``A_k(phi) = a cos(phi + delta)``: it
    peaks at ``phi = -delta`` (mod 360). See :func:`phase_of_maximum`.
    """
    if phases_deg is None:
        phases_deg = np.arange(0.0, 360.0, 10.0)
    phases_deg = np.asarray(phases_deg, dtype=float)
    in_phase, quadrature, period = _quadrature_components(ir_period, harmonics, period)
    phi = np.deg2rad(phases_deg)[None, :, None]
    demodulated = in_phase[:, None, :] * np.cos(phi) + quadrature[:, None, :] * np.sin(phi)
    return xr.DataArray(
        demodulated,
        dims=("harmonic", "phase", "wavenumber"),
        coords={
            "harmonic": np.asarray(harmonics, dtype=int),
            "phase": phases_deg,
            "wavenumber": ir_period.ds.coords["wavenumber"].values,
        },
        name="demodulated",
        attrs={
            "period": period,
            "phase_unit": "deg",
            "wavenumber_unit": "m^-1",
            "data_type": ir_period.ds.attrs.get("data_type", ""),
            "n_cycles_averaged": int(ir_period.ds.attrs.get("n_cycles_averaged", 0)),
        },
    )


def phase_of_maximum(
    ir_period: IRData,
    harmonics: Sequence[int] = (1, 2, 3),
    period: Optional[float] = None,
) -> xr.Dataset:
    """Amplitude and phase angle of maximum of each harmonic, in closed form.

    The maximum of ``A_k(phi) = I_k cos(phi) + Q_k sin(phi)`` over ``phi`` is
    ``sqrt(I_k^2 + Q_k^2)`` at ``phi = atan2(Q_k, I_k)``, so no phase grid
    is needed.

    Parameters
    ----------
    ir_period : IRData
        One averaged period, see :func:`demodulate`.
    harmonics : sequence of int, optional
        Harmonics ``k`` (default is ``(1, 2, 3)``).
    period : float or None, optional
        Modulation period ``T`` in seconds. ``None`` reads
        ``ds.attrs['modulation_period']``.

    Returns
    -------
    xarray.Dataset
        Variables ``amplitude`` and ``phase_max`` (degrees in ``[0, 360)``),
        both with dims ``('harmonic', 'wavenumber')``.

    Notes
    -----
    ``phase_max`` is where the band is most positive. A band that goes down
    when the stimulus goes up has ``phase_max`` shifted by 180 degrees from
    one that goes up with the same timing: take the sign from the raw traces
    before reading a phase difference as a time lag.
    """
    in_phase, quadrature, period = _quadrature_components(ir_period, harmonics, period)
    dims = ("harmonic", "wavenumber")
    return xr.Dataset(
        {
            "amplitude": (dims, np.hypot(in_phase, quadrature)),
            "phase_max": (dims, np.mod(np.rad2deg(np.arctan2(quadrature, in_phase)), 360.0)),
        },
        coords={"harmonic": np.asarray(harmonics, dtype=int), "wavenumber": ir_period.ds.coords["wavenumber"].values},
        attrs={"period": period, "phase_unit": "deg"},
    )
