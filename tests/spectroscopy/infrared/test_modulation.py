import numpy as np
import pytest

from phd_parser.spectroscopy.infrared import IRData
from phd_parser.spectroscopy.infrared import modulation
from phd_parser.spectroscopy import SpectroscopyData

WAVENUMBER_PER_CM = np.linspace(1000.0, 3000.0, 41)
PERIOD = 600.0
DT = 3.0


def lagged_response(t, lag):
    """Smooth periodic response with a known time lag behind the stimulus."""
    phase = 2 * np.pi * (t - lag) / PERIOD
    return np.sin(phase) + 0.3 * np.sin(2 * phase)


def make_modulated(n_cycles=6, lag=45.0, drift_per_s=0.0, jitter=None, noise=0.0, seed=0):
    """Band at every wavenumber, amplitude scaled per wavenumber, starts at tos 0.

    With ``jitter`` each cycle starts at its own logged time; the signal at any
    moment follows the most recent start, as it would in a real switching run.
    """
    rng = np.random.default_rng(seed)
    tos = np.arange(0.0, n_cycles * PERIOD + 30.0, DT)
    starts = np.arange(n_cycles) * PERIOD
    if jitter is not None:
        starts = starts + jitter
    last_start = starts[np.clip(np.searchsorted(starts, tos, side="right") - 1, 0, None)]
    signal = lagged_response(tos - last_start, lag)
    amplitude = np.linspace(0.5, 1.5, WAVENUMBER_PER_CM.size)
    values = signal[:, None] * amplitude[None, :] + drift_per_s * tos[:, None]
    values = values + noise * rng.standard_normal(values.shape)
    ir = IRData.from_arrays(WAVENUMBER_PER_CM, values, tos=tos, data_type="absorbance")
    return ir, starts, amplitude


def test_fold_average_recovers_lagged_signal():
    lag = 45.0
    ir, starts, amplitude = make_modulated(lag=lag)
    folded = modulation.fold_cycles(ir, starts, PERIOD, n_discard=2)

    assert folded.dims == ("cycle", "time", "wavenumber")
    # 6 starts, the last one runs off the end of the data only if incomplete; all 6 complete here
    assert folded.sizes["cycle"] == 4
    np.testing.assert_allclose(folded.coords["cycle"], starts[2:])

    averaged = modulation.average_cycles(folded)
    assert averaged.ds.attrs["n_cycles_averaged"] == 4
    assert averaged.ds.attrs["modulation_period"] == PERIOD
    expected = lagged_response(averaged.tos, lag)[:, None] * amplitude[None, :]
    np.testing.assert_allclose(averaged.values, expected, atol=1e-9)


def test_fold_uses_logged_starts_so_jitter_does_not_smear():
    jitter = np.array([0.0, 12.0, -9.0, 15.0, -12.0, 6.0])
    ir, starts, amplitude = make_modulated(jitter=jitter)
    nominal = np.arange(6) * PERIOD

    def max_error(cycle_starts):
        averaged = modulation.average_cycles(modulation.fold_cycles(ir, cycle_starts, PERIOD, n_discard=0))
        # the tail of a period can already belong to the next (early) cycle; compare before it
        keep = averaged.tos < PERIOD - 30.0
        expected = lagged_response(averaged.tos, 45.0)[:, None] * amplitude[None, :]
        return np.abs(averaged.values - expected)[keep].max()

    assert max_error(starts) < 5e-3
    assert max_error(nominal) > 10 * max_error(starts)


def test_fold_skips_incomplete_cycles():
    ir, starts, _ = make_modulated(n_cycles=4)
    extra = np.append(starts, starts[-1] + PERIOD)  # starts after the data ends
    folded = modulation.fold_cycles(ir, extra, PERIOD, n_discard=0)
    assert folded.sizes["cycle"] == 4


def test_fold_raises_when_nothing_left():
    ir, starts, _ = make_modulated(n_cycles=2)
    with pytest.raises(ValueError):
        modulation.fold_cycles(ir, starts, PERIOD, n_discard=2)


def test_correct_drift_removes_ramp_and_keeps_modulation():
    ir, starts, amplitude = make_modulated(drift_per_s=1e-3)
    corrected = ir.correct_drift(starts, PERIOD, order=1)
    averaged = modulation.average_cycles(modulation.fold_cycles(corrected, starts, PERIOD, n_discard=0))
    # the averaged period keeps its shape; only a constant level is left from the ramp
    expected = lagged_response(averaged.tos, 45.0)[:, None] * amplitude[None, :]
    residual = averaged.values - expected
    np.testing.assert_allclose(residual - residual.mean(axis=0), 0.0, atol=1e-6)
    # every cycle's mean is now the same: no trend left
    folded = modulation.fold_cycles(corrected, starts, PERIOD, n_discard=0)
    cycle_means = folded.mean(dim="time").values
    np.testing.assert_allclose(cycle_means - cycle_means[0], 0.0, atol=1e-6)
    # the removed drift is stored in the baseline
    np.testing.assert_allclose(corrected.data_unbaselined, ir.values, atol=1e-9)


def test_mask_wavenumber_ranges_sets_nan_only_inside():
    ir, _, _ = make_modulated(n_cycles=2)
    masked = ir.mask_wavenumber_ranges([(2400, 2200)])
    inside = (WAVENUMBER_PER_CM >= 2200) & (WAVENUMBER_PER_CM <= 2400)
    assert np.isnan(masked.values[:, inside]).all()
    np.testing.assert_array_equal(masked.values[:, ~inside], ir.values[:, ~inside])
    assert masked.ds.attrs["masked_ranges_cm"] == [2200.0, 2400.0]


def test_find_and_drop_spike_scans():
    # noise comparable to the per-scan change of the signal, as in real data
    ir, starts, _ = make_modulated(n_cycles=3, noise=0.02)
    values = ir.values.copy()
    spikes = [100, 101, 350]
    values[spikes] += 0.2
    spiky = IRData.from_arrays(WAVENUMBER_PER_CM, values, tos=ir.tos, data_type="absorbance")
    flagged = spiky.find_spike_scans()
    assert set(np.where(flagged)[0]) == set(spikes)
    cleaned = spiky.drop_scans(flagged)
    assert cleaned.shape[0] == ir.shape[0] - 3
    assert cleaned.ds.attrs["n_scans_dropped"] == 3


def one_period(signal_of_t, n_points=1200, period=PERIOD):
    """One averaged period on the grid average_cycles produces, same signal at every wavenumber."""
    time = np.arange(n_points) * period / n_points
    values = signal_of_t(time)[:, None] * np.ones(WAVENUMBER_PER_CM.size)[None, :]
    ir = IRData.from_arrays(WAVENUMBER_PER_CM, values, tos=time, data_type="absorbance")
    return SpectroscopyData(ds=ir.ds.assign_attrs(modulation_period=period))


def test_demodulate_recovers_known_phase_lag_and_amplitude():
    omega = 2 * np.pi / PERIOD
    lag_deg, lag2_deg = 70.0, 25.0
    ir = one_period(
        lambda t: 0.8 * np.sin(omega * t - np.deg2rad(lag_deg)) + 0.3 * np.sin(2 * omega * t - np.deg2rad(lag2_deg))
    )

    demodulated = modulation.demodulate(ir)
    assert demodulated.dims == ("harmonic", "phase", "wavenumber")
    np.testing.assert_array_equal(demodulated.phase, np.arange(0, 360, 10))

    # A_k(phi) = a cos(phi + lag): the k = 1 peak sits at -70 = 290 deg with height 0.8
    k1 = demodulated.sel(harmonic=1).isel(wavenumber=0)
    assert float(k1.phase[int(np.argmax(k1.data))]) == 290.0
    np.testing.assert_allclose(k1.values, 0.8 * np.cos(np.deg2rad(k1.phase + lag_deg)), atol=1e-12)
    k2 = demodulated.sel(harmonic=2).isel(wavenumber=0)
    np.testing.assert_allclose(k2.values, 0.3 * np.cos(np.deg2rad(k2.phase + lag2_deg)), atol=1e-12)
    np.testing.assert_allclose(demodulated.sel(harmonic=3).values, 0.0, atol=1e-12)

    peak = modulation.phase_of_maximum(ir)
    np.testing.assert_allclose(peak["phase_max"].sel(harmonic=1), 360 - lag_deg, atol=1e-9)
    np.testing.assert_allclose(peak["amplitude"].sel(harmonic=1), 0.8, atol=1e-12)
    np.testing.assert_allclose(peak["phase_max"].sel(harmonic=2), 360 - lag2_deg, atol=1e-9)


def test_phase_difference_between_two_first_order_responses():
    # periodic steady state of dy/dt = (u - y) / tau for a sine stimulus u: phase lag atan(w tau)
    omega = 2 * np.pi / PERIOD
    tau_fast, tau_slow = 5.0, 60.0

    def first_order(tau):
        return lambda t: np.sin(omega * t - np.arctan(omega * tau)) / np.sqrt(1 + (omega * tau) ** 2)

    phase_fast = float(modulation.phase_of_maximum(one_period(first_order(tau_fast)))["phase_max"].sel(harmonic=1)[0])
    phase_slow = float(modulation.phase_of_maximum(one_period(first_order(tau_slow)))["phase_max"].sel(harmonic=1)[0])
    # the slower response peaks at a smaller phase angle; the difference is its extra lag
    expected = np.rad2deg(np.arctan(omega * tau_slow) - np.arctan(omega * tau_fast))
    np.testing.assert_allclose(np.mod(phase_fast - phase_slow, 360), expected, atol=1e-9)


def test_demodulate_ignores_constant_offset_and_keeps_nan():
    omega = 2 * np.pi / PERIOD
    ir = one_period(lambda t: 5.0 + np.sin(omega * t))
    values = ir.values.copy()
    values[:, 3] = np.nan
    ir = SpectroscopyData(ds=ir.ds.assign(data=(ir.ds["data"].dims, values)))
    demodulated = modulation.demodulate(ir)
    assert np.isnan(demodulated.isel(wavenumber=3)).all()
    np.testing.assert_allclose(demodulated.sel(harmonic=1).isel(wavenumber=0).max(), 1.0, atol=1e-12)


def test_demodulate_rejects_data_that_is_not_one_period():
    ir, _, _ = make_modulated(n_cycles=2)
    with pytest.raises(ValueError):
        modulation.demodulate(ir, period=PERIOD)


def test_fold_then_demodulate_end_to_end():
    lag = 45.0
    ir, starts, amplitude = make_modulated(lag=lag)
    peak = modulation.phase_of_maximum(modulation.average_cycles(modulation.fold_cycles(ir, starts, PERIOD)))
    # lagged_response is sin(w (t - lag)) + 0.3 sin(2 w (t - lag)): k = 1 lag of 360 lag / T degrees
    np.testing.assert_allclose(peak["phase_max"].sel(harmonic=1), 360 - 360 * lag / PERIOD, atol=1e-6)
    np.testing.assert_allclose(peak["phase_max"].sel(harmonic=2), 360 - 2 * 360 * lag / PERIOD, atol=1e-6)
    np.testing.assert_allclose(peak["amplitude"].sel(harmonic=1), amplitude, rtol=1e-6)
    np.testing.assert_allclose(peak["amplitude"].sel(harmonic=2), 0.3 * amplitude, rtol=1e-6)


def test_average_cycles_keeps_the_history_of_the_series():
    ir, starts, _ = make_modulated(drift_per_s=1e-4)
    corrected = ir.correct_drift(starts, PERIOD, order=1)
    period = modulation.average_cycles(modulation.fold_cycles(corrected, starts, PERIOD, n_discard=0))
    steps = [entry["step"] for entry in period.history]
    assert steps[:2] == ["from_arrays", "correct_drift"] and steps[-1] == "average_cycles"
