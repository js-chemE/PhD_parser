"""Modulation excitation spectroscopy (MES): folding, averaging and demodulating a period.

The steps that return a new spectrum (``mask_wavenumber_ranges``, ``drop_scans``,
``drop_spike_scans``, ``find_spike_scans``, ``correct_drift``) are methods of
``SpectroscopyData``, recorded in its history; this module keeps what works on
the folded ``xarray.DataArray`` of cycles.

Preprocessing and cycle folding for periodic-stimulation experiments (e.g.
switching the feed between H2 and CO2 + H2). Every function takes an
``SpectroscopyData`` and returns a new object; nothing is modified in place, in keeping
with the ``SpectroscopyData`` convention.

Cycle times are plain ``tos`` values in seconds (the start of each period,
usually the moment the stimulus is switched on), so this module does not
depend on how the switches were logged.
"""

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import xarray as xr

from phd_parser.history import HISTORY_KEY, decode_attr, encode_attr, records_step
from phd_parser.spectroscopy._baseline import full_cycle_starts as _full_cycle_starts
from phd_parser.spectroscopy.core import SpectroscopyData

logger = logging.getLogger(__name__)

def fold_cycles(
    ir: SpectroscopyData,
    cycle_starts: Sequence[float],
    period: float,
    n_discard: int = 2,
    time_step: Optional[float] = None,
) -> xr.DataArray:
    """Cut the time series into periods and put them on a common time grid.

    Parameters
    ----------
    ir : SpectroscopyData
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
        ``wavenumber`` is in m⁻¹ as in ``SpectroscopyData``. Attrs carry ``period`` and
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
            "history": encode_attr(ir.history),  # carried to average_cycles, JSON so the array stays writable
        },
    )


@records_step
def average_cycles(folded: xr.DataArray) -> SpectroscopyData:
    """Average folded cycles into one period.

    Parameters
    ----------
    folded : xarray.DataArray
        Output of :func:`fold_cycles`.

    Returns
    -------
    SpectroscopyData
        One period, ``tos`` is the time within the period in seconds and no
        ``tos_start`` is set. ``ds.attrs`` carries ``modulation_period`` and
        ``n_cycles_averaged``.
    """
    mean = folded.mean(dim="cycle").values
    ir = SpectroscopyData.from_arrays(
        folded.coords["wavenumber"].values / 100.0,
        mean,
        tos=folded.coords["time"].values,
        data_type=folded.attrs.get("data_type") or "absorbance",
    )
    ds_new = ir.ds.copy()
    ds_new.attrs = {
        **ir.ds.attrs,
        HISTORY_KEY: list(decode_attr(folded.attrs.get("history", [])) or []),  # the parent's steps, then this one
        "modulation_period": folded.attrs["period"],
        "n_cycles_averaged": int(folded.sizes["cycle"]),
        "n_cycles_discarded": folded.attrs.get("n_discarded", 0),
    }
    return SpectroscopyData(ds=ds_new)


def _quadrature_components(
    ir_period: SpectroscopyData,
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
    ir_period: SpectroscopyData,
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
    ir_period : SpectroscopyData
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
        ``wavenumber`` in m⁻¹ as in ``SpectroscopyData``. A NaN (masked) wavenumber
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
    ir_period: SpectroscopyData,
    harmonics: Sequence[int] = (1, 2, 3),
    period: Optional[float] = None,
) -> xr.Dataset:
    """Amplitude and phase angle of maximum of each harmonic, in closed form.

    The maximum of ``A_k(phi) = I_k cos(phi) + Q_k sin(phi)`` over ``phi`` is
    ``sqrt(I_k^2 + Q_k^2)`` at ``phi = atan2(Q_k, I_k)``, so no phase grid
    is needed.

    Parameters
    ----------
    ir_period : SpectroscopyData
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
