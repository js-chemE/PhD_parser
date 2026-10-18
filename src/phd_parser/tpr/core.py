"""Temperature-programmed reduction: :class:`TPRData`, one run of the LabVIEW TPR rig.

A TPR run is a bed temperature and a few detector signals, all against time:
the furnace ramps at a set rate, the TCD on the hydrogen line sees the
consumption (or release) of hydrogen as an excursion of its bridge voltage.
``TPRData`` stores one run as an ``xarray.Dataset`` over the ``time``
dimension (seconds since the start of writing): ``temperature`` in °C and
one variable per detector signal (``tcd_h2``, ``tcd_o2``, ``fid`` when the
export has it) in mV, exactly as the export gives them.

The sign of a TCD signal is the bridge wiring's, not physics': whether a
hydrogen consumption appears as a positive or a negative excursion depends
on which side of the bridge the sample gas passes and on the polarity the
LabVIEW program records. In the Urakawa rig's exports so far the reduction
of the molybdenum oxides shows as a *positive* excursion of ``tcd_h2``.
This class never flips a signal on read; :meth:`TPRData.scale` with
``factor=-1`` is the explicit way to do so.

Reader: :meth:`TPRData.from_labview_tcd` over
:mod:`phd_parser.tpr.labview_tcd`. Every processing step returns a new
object and is recorded in the history.
"""

import logging
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, field_validator

from phd_parser._selection import EmptyWindow, window_mask
from phd_parser.history import HistoryMixin, decode_attrs, encode_attrs, records_step

logger = logging.getLogger(__name__)

TIME = "time"
TEMPERATURE = "temperature"
TIMESTAMP = "timestamp"
BASELINE_SUFFIX = "_baseline"
TEMPERATURE_UNIT = "°C"
SIGNAL_UNIT = "mV"

BaselineMethod = Literal["linear", "constant"]


class TPRData(HistoryMixin, BaseModel):
    """Immutable wrapper around one TPR run: temperature and detector signals over time.

    Attributes
    ----------
    ds : xr.Dataset
        The ``time`` dimension and coordinate (seconds since the start of
        writing, float); ``temperature`` (°C, ``TC 1``) and one variable per
        detector signal (``tcd_h2``, ``tcd_o2``, ``fid``, ... in mV, raw
        sign), each with ``units`` and ``source_column`` in its attrs; an
        optional ``timestamp`` coordinate (per-row wall clock, variant-2
        exports); ``<signal>_baseline`` after :meth:`subtract_baseline`.
        ``ds.attrs`` hold ``technique`` (``'tpr'``), ``sample_name``,
        ``sample_weight_mg``, ``start_time`` (ISO string, timezone-naive
        wall clock of the rig), ``comment``, ``username``, ``source_file``,
        ``format_variant``, ``columns`` (the original column names) and the
        ``history``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True, ignored_types=(cached_property,))

    ds: xr.Dataset = Field(description="Dataset with 'temperature' and the detector signals over 'time' (s).")

    @field_validator("ds", mode="before")
    @classmethod
    def validate_ds(cls, v: Any) -> xr.Dataset:
        """Check the layout of ``ds``: a ``time`` dimension and 1-D ``temperature`` and signals over it."""
        if not isinstance(v, xr.Dataset):
            raise TypeError(f"'ds' must be an xr.Dataset, got {type(v)}")
        if TIME not in v.dims:
            raise ValueError(f"Dataset must have a '{TIME}' dimension")
        if TEMPERATURE not in v.data_vars:
            raise ValueError(f"Dataset must contain a '{TEMPERATURE}' variable")
        for name in v.data_vars:
            if tuple(v[name].dims) != (TIME,):
                raise ValueError(f"'{name}' must be 1-D over '{TIME}', got dims {v[name].dims}")
        return v

    # ----------------------------------------------------------------
    # Accessors
    # ----------------------------------------------------------------

    @property
    def time(self) -> npt.NDArray[np.float64]:
        """Time axis in s since the start of writing."""
        return self.ds.coords[TIME].values

    @property
    def temperature(self) -> npt.NDArray[np.float64]:
        """Bed temperature (``TC 1``) in °C."""
        return self.ds[TEMPERATURE].values

    @property
    def signals(self) -> list[str]:
        """Names of the detector signal variables (everything but the temperature and the baselines)."""
        return [str(name) for name in self.ds.data_vars
                if name != TEMPERATURE and not str(name).endswith(BASELINE_SUFFIX)]

    @property
    def n_points(self) -> int:
        """Number of samples on the time axis."""
        return self.ds.sizes[TIME]

    @property
    def sample_name(self) -> Optional[str]:
        """Sample name from the header."""
        return self.ds.attrs.get("sample_name")

    @property
    def sample_weight_mg(self) -> Optional[float]:
        """Sample weight in mg from the header, or ``None``."""
        value = self.ds.attrs.get("sample_weight_mg")
        return None if value is None else float(value)

    @property
    def start_time(self) -> Optional[pd.Timestamp]:
        """When writing started, as the rig's wall clock showed it (timezone-naive, Europe/Amsterdam)."""
        raw = self.ds.attrs.get("start_time")
        return None if raw is None else pd.Timestamp(raw)

    @property
    def comment(self) -> Optional[str]:
        """The comment line of the header (usually the programme: final temperature, ramp, hold)."""
        return self.ds.attrs.get("comment")

    @property
    def timestamps(self) -> Optional[npt.NDArray[np.datetime64]]:
        """Per-row wall-clock stamps (variant-2 exports), or ``None``."""
        return self.ds.coords[TIMESTAMP].values if TIMESTAMP in self.ds.coords else None

    def get_signal(self, name: str) -> xr.DataArray:
        """One detector signal (or baseline) as a DataArray over ``time``.

        Parameters
        ----------
        name : str
            A variable name, e.g. ``"tcd_h2"``.

        Returns
        -------
        xr.DataArray
            The variable, units in its attrs.

        Raises
        ------
        KeyError
            If there is no such variable; the message lists the signals.
        """
        if name not in self.ds.data_vars or name == TEMPERATURE:
            raise KeyError(f"No signal {name!r}; the signals are {self.signals}")
        return self.ds[name]

    def integrate(self, signal: str, min_s: Optional[float] = None, max_s: Optional[float] = None) -> float:
        """Area under a signal over a time window (trapezoid rule, NaN samples skipped).

        Parameters
        ----------
        signal : str
            The signal name.
        min_s, max_s : float or None, optional
            Inclusive bounds in s; ``None`` leaves that side open.

        Returns
        -------
        float
            The area in the signal's units times s (``mV s`` for a raw
            signal). The sign follows the signal's.

        Raises
        ------
        ValueError
            If fewer than two finite samples fall inside the window.
        """
        values = self.get_signal(signal).values
        mask = window_mask(self.time, min_s, max_s, what="sample", axis=TIME, unit="s", empty="raise")
        mask &= np.isfinite(values)
        if mask.sum() < 2:
            raise ValueError(f"Fewer than two finite samples of {signal!r} in [{min_s}, {max_s}] s")
        return float(np.trapezoid(values[mask], x=self.time[mask]))

    def heating_rate(self, window_s: float = 60.0, threshold_K_per_min: float = 1.0) -> float:
        """Median heating rate of the ramp in K min^-1.

        The thermocouple reads to 0.1 K once a second, so a point-wise
        derivative is noise; the temperature is first averaged over a
        centred window of ``window_s`` and differentiated, and the median
        is taken over the samples where the rate exceeds the threshold
        (the ramp, not the dwell or the cooling).

        Parameters
        ----------
        window_s : float, optional
            Averaging window in s (default is 60).
        threshold_K_per_min : float, optional
            A sample counts as ramp above this rate (default is 1).

        Returns
        -------
        float
            The median rate in K min^-1; NaN with a warning if no sample
            rises faster than the threshold.
        """
        time = self.time
        temperature = self.temperature
        finite = np.isfinite(time) & np.isfinite(temperature)
        if finite.sum() < 3:
            logger.warning("heating_rate needs at least three finite samples; returning NaN")
            return float("nan")
        time, temperature = time[finite], temperature[finite]
        spacing = float(np.median(np.diff(time))) if time.size > 1 else 1.0
        n_window = max(1, int(round(window_s / spacing)))
        n_window += 1 - n_window % 2  # odd, so the window is centred
        smoothed = pd.Series(temperature).rolling(n_window, center=True, min_periods=1).mean().to_numpy()
        rate = np.gradient(smoothed, time) * 60.0
        rising = rate[rate > threshold_K_per_min]
        if rising.size == 0:
            logger.warning("No sample heats faster than %g K min^-1; returning NaN", threshold_K_per_min)
            return float("nan")
        return float(np.median(rising))

    # ----------------------------------------------------------------
    # Immutable processing
    # ----------------------------------------------------------------

    def _new(self, ds: xr.Dataset) -> "TPRData":
        return TPRData(ds=ds)

    def _apply_to_signal(self, signal: str, function: Callable[[npt.NDArray], npt.NDArray],
                         with_baseline: bool = True, **attrs: Any) -> "TPRData":
        """Apply ``function`` to a signal (and its baseline) and update the variable attrs."""
        self.get_signal(signal)
        ds = self.ds.copy()
        names = [signal]
        if with_baseline and signal + BASELINE_SUFFIX in ds:
            names.append(signal + BASELINE_SUFFIX)
        for name in names:
            variable_attrs = {**ds[name].attrs, **attrs}
            ds[name] = xr.DataArray(np.asarray(function(ds[name].values), dtype=float), dims=[TIME],
                                    coords={TIME: ds.coords[TIME]}, attrs=variable_attrs)
        return self._new(ds)

    @records_step
    def select_time_range(self, min_s: Optional[float] = None, max_s: Optional[float] = None, *,
                          empty: EmptyWindow = "warn") -> "TPRData":
        """Return a new instance restricted to a time window (inclusive bounds).

        Parameters
        ----------
        min_s, max_s : float or None, optional
            Bounds in s since the start; ``None`` leaves that side open.
        empty : {"warn", "raise"}, optional
            A window with no sample logs a warning and returns the empty
            selection (default), or raises.

        Returns
        -------
        TPRData
            New instance; every variable is cut to the window.
        """
        mask = window_mask(self.time, min_s, max_s, what="sample", axis=TIME, unit="s", empty=empty)
        return self._new(self.ds.isel({TIME: mask}))

    @records_step
    def select_temperature_range(self, min_C: Optional[float] = None, max_C: Optional[float] = None, *,
                                 empty: EmptyWindow = "warn") -> "TPRData":
        """Return a new instance with the samples whose temperature lies in a window, in time order.

        A run passes a temperature twice (heating and cooling), so the
        selection is a mask, not a slice: both passes are kept, in time
        order, with a gap between them.

        Parameters
        ----------
        min_C, max_C : float or None, optional
            Inclusive bounds in °C; ``None`` leaves that side open.
        empty : {"warn", "raise"}, optional
            A window with no sample logs a warning and returns the empty
            selection (default), or raises.

        Returns
        -------
        TPRData
            New instance.
        """
        mask = window_mask(self.temperature, min_C, max_C, what="sample", axis=TEMPERATURE,
                           unit=TEMPERATURE_UNIT, empty=empty)
        return self._new(self.ds.isel({TIME: mask}))

    @records_step
    def smooth_moving(self, window_size: int = 5) -> "TPRData":
        """Apply a centred moving average to every detector signal; the temperature is left alone.

        Parameters
        ----------
        window_size : int, optional
            Number of samples in the window (default is 5). The window
            shrinks at the edges and skips NaN samples, so a gap does not
            spread (a lone NaN is filled from its neighbours).

        Returns
        -------
        TPRData
            New instance with the smoothed signals.

        Raises
        ------
        ValueError
            If ``window_size`` is less than 1.
        """
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        result = self
        for name in self.signals:
            result = result._apply_to_signal(
                name,
                lambda values: pd.Series(values).rolling(window_size, center=True, min_periods=1).mean().to_numpy(dtype=float),
                with_baseline=False, smooth_moving_window=window_size,
            )
        return result

    @records_step
    def scale(self, signal: str, factor: float) -> "TPRData":
        """Multiply a signal (and its baseline) by a factor; ``factor=-1`` flips a TCD whose bridge reads inverted.

        Parameters
        ----------
        signal : str
            The signal name.
        factor : float
            The multiplier.

        Returns
        -------
        TPRData
            New instance; the variable's ``scaled_by`` attr holds the
            cumulative factor.
        """
        previous = float(self.get_signal(signal).attrs.get("scaled_by", 1.0))
        return self._apply_to_signal(signal, lambda values: values * float(factor), scaled_by=previous * float(factor))

    @records_step
    def per_gram(self, signal: str) -> "TPRData":
        """Divide a signal (and its baseline) by the sample mass in g.

        Parameters
        ----------
        signal : str
            The signal name.

        Returns
        -------
        TPRData
            New instance; the variable's ``units`` gain ``g^-1``
            (``mV g^-1`` for a raw signal).

        Raises
        ------
        ValueError
            If no sample weight is stored, or the signal is already per gram.
        """
        units = str(self.get_signal(signal).attrs.get("units", SIGNAL_UNIT))
        if units.endswith("g^-1"):
            raise ValueError(f"{signal!r} is already per gram ({units})")
        if self.sample_weight_mg is None or self.sample_weight_mg <= 0:
            raise ValueError("per_gram needs a positive sample_weight_mg in the attrs")
        mass_g = self.sample_weight_mg / 1000.0
        return self._apply_to_signal(signal, lambda values: values / mass_g, units=f"{units} g^-1",
                                     sample_weight_mg=self.sample_weight_mg)

    @records_step
    def subtract_baseline(
        self,
        signal: str,
        method: BaselineMethod = "linear",
        time_window_s: Optional[tuple[float, float]] = None,
        anchor_points_s: Optional[Sequence[float]] = None,
        anchor_half_width_s: float = 30.0,
    ) -> "TPRData":
        """Subtract a linear or constant baseline from a signal and keep it as ``<signal>_baseline``.

        The baseline is defined either by one time window (a least-squares
        line through, or the mean of, the samples inside it) or by anchor
        points (a line through, or the mean of, the signal averaged over
        ``anchor_half_width_s`` around each point). Anchors before and after
        the reduction peak are the usual choice for a TPR; a window on the
        isothermal start is the usual choice for a drifting TCD.

        Parameters
        ----------
        signal : str
            The signal name.
        method : {"linear", "constant"}, optional
            Shape of the baseline (default is ``"linear"``).
        time_window_s : tuple of float or None, optional
            ``(t0, t1)`` in s: the samples the baseline is fitted to.
        anchor_points_s : sequence of float or None, optional
            Times in s the baseline passes through; at least two for
            ``"linear"``, one for ``"constant"``.
        anchor_half_width_s : float, optional
            Half width in s of the average around each anchor (default
            is 30).

        Returns
        -------
        TPRData
            New instance with the baseline removed from the signal and
            stored in ``<signal>_baseline`` (the total removed so far, if a
            baseline had been subtracted before); the signal's
            ``baseline_method`` attr names the method.

        Raises
        ------
        ValueError
            For an unknown method, when not exactly one of ``time_window_s``
            and ``anchor_points_s`` is given, or when the fit has too few
            samples.
        """
        if method not in ("linear", "constant"):
            raise ValueError(f"method must be 'linear' or 'constant', got {method!r}")
        if (time_window_s is None) == (anchor_points_s is None):
            raise ValueError("Give exactly one of time_window_s and anchor_points_s")
        time = self.time
        values = self.get_signal(signal).values
        if time_window_s is not None:
            lower, upper = time_window_s
            mask = window_mask(time, lower, upper, what="sample", axis=TIME, unit="s", empty="raise") & np.isfinite(values)
            fit_time, fit_values = time[mask], values[mask]
        else:
            anchors = np.asarray(anchor_points_s, dtype=float).ravel()
            fit_values_list = []
            for anchor in anchors:
                mask = window_mask(time, anchor - anchor_half_width_s, anchor + anchor_half_width_s,
                                   what="sample", axis=TIME, unit="s", empty="raise") & np.isfinite(values)
                if not mask.any():
                    raise ValueError(f"No finite sample of {signal!r} within {anchor_half_width_s} s of {anchor} s")
                fit_values_list.append(float(np.mean(values[mask])))
            fit_time, fit_values = anchors, np.asarray(fit_values_list)
        if method == "constant":
            if fit_values.size < 1:
                raise ValueError(f"No finite sample of {signal!r} to take the baseline from")
            baseline = np.full(time.shape, float(np.mean(fit_values)))
        else:
            if fit_values.size < 2 or np.ptp(fit_time) == 0:
                raise ValueError("A linear baseline needs at least two distinct times")
            slope, intercept = np.polyfit(fit_time, fit_values, 1)
            baseline = slope * time + intercept
        ds = self.ds.copy()
        baseline_name = signal + BASELINE_SUFFIX
        total = baseline + (ds[baseline_name].values if baseline_name in ds else 0.0)
        signal_attrs = {**ds[signal].attrs, "baseline_method": method}
        ds[signal] = xr.DataArray(values - baseline, dims=[TIME], coords={TIME: ds.coords[TIME]}, attrs=signal_attrs)
        ds[baseline_name] = xr.DataArray(total, dims=[TIME], coords={TIME: ds.coords[TIME]},
                                         attrs={"units": ds[signal].attrs.get("units", SIGNAL_UNIT),
                                                "baseline_method": method, "baseline_of": signal})
        return self._new(ds)

    # ----------------------------------------------------------------
    # Export
    # ----------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Temperature, every signal and baseline (and the timestamps, if any) as a table indexed by time in s."""
        columns: dict[str, Any] = {}
        if TIMESTAMP in self.ds.coords:
            columns[TIMESTAMP] = self.timestamps
        columns["temperature_C"] = self.temperature
        for name in self.ds.data_vars:
            if name != TEMPERATURE:
                columns[str(name)] = self.ds[name].values
        return pd.DataFrame(columns, index=pd.Index(self.time, name="time_s"))

    def to_netcdf(self, filepath: Union[str, Path]) -> None:
        """Save to NetCDF4 (through h5netcdf), history included; :meth:`from_netcdf` reloads it.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Destination; an existing file is overwritten with a warning.
        """
        filepath = Path(filepath)
        if filepath.exists():
            logger.warning(f"Overwriting existing file: {filepath}")
        ds = self.ds.copy()
        ds.attrs = encode_attrs(dict(self.ds.attrs))
        for name in ds.variables:
            ds[name].attrs = encode_attrs(dict(ds[name].attrs))
        ds.to_netcdf(filepath, engine="h5netcdf")

    # ----------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------

    @classmethod
    def _build(
        cls,
        time_s: npt.NDArray,
        temperature_C: npt.NDArray,
        signals: dict[str, npt.NDArray],
        attrs: dict[str, Any],
        *,
        timestamp: Optional[npt.NDArray] = None,
        units: Optional[dict[str, str]] = None,
        source_columns: Optional[dict[str, str]] = None,
    ) -> "TPRData":
        time = np.asarray(time_s, dtype=float)
        temperature = np.asarray(temperature_C, dtype=float)
        if time.ndim != 1 or temperature.shape != time.shape:
            raise ValueError(f"temperature_C {temperature.shape} must be 1-D and match time_s {time.shape}")
        coords: dict[str, Any] = {TIME: time}
        if timestamp is not None:
            stamps = np.asarray(timestamp)
            if stamps.shape != time.shape:
                raise ValueError(f"timestamp {stamps.shape} must match time_s {time.shape}")
            coords[TIMESTAMP] = (TIME, stamps.astype("datetime64[ns]"))
        units = units or {}
        source_columns = source_columns or {}
        variables: dict[str, Any] = {
            TEMPERATURE: xr.DataArray(temperature, dims=[TIME], attrs={
                "units": TEMPERATURE_UNIT, "long_name": "bed temperature",
                **({"source_column": source_columns[TEMPERATURE]} if TEMPERATURE in source_columns else {})}),
        }
        for name, array in signals.items():
            if name == TEMPERATURE or name.endswith(BASELINE_SUFFIX):
                raise ValueError(f"{name!r} is not a valid signal name")
            array = np.asarray(array, dtype=float)
            if array.shape != time.shape:
                raise ValueError(f"{name} {array.shape} must match time_s {time.shape}")
            variable_attrs = {"units": units.get(name, SIGNAL_UNIT)}
            if name in source_columns:
                variable_attrs["source_column"] = source_columns[name]
            variables[name] = xr.DataArray(array, dims=[TIME], attrs=variable_attrs)
        clean: dict[str, Any] = {"technique": "tpr"}
        for key, value in attrs.items():
            if value is None:
                continue
            if isinstance(value, (pd.Timestamp, np.datetime64)):
                value = pd.Timestamp(value).isoformat()
            clean[key] = value
        ds = xr.Dataset(variables, coords=coords, attrs=clean)
        return cls(ds=ds)

    @classmethod
    @records_step
    def from_arrays(
        cls,
        time_s: npt.NDArray,
        temperature_C: npt.NDArray,
        signals: dict[str, npt.NDArray],
        *,
        timestamp: Optional[npt.NDArray] = None,
        units: Optional[dict[str, str]] = None,
        **attrs: Any,
    ) -> "TPRData":
        """Build a run from raw arrays.

        Parameters
        ----------
        time_s : numpy.ndarray
            1-D time axis in s.
        temperature_C : numpy.ndarray
            Bed temperature in °C, same length.
        signals : dict of str to numpy.ndarray
            Detector signals by name (``"tcd_h2"``, ...), each the same
            length; mV unless ``units`` says otherwise.
        timestamp : numpy.ndarray or None, optional
            Per-row wall-clock stamps (``datetime64``), same length.
        units : dict of str to str or None, optional
            Units per signal name; ``"mV"`` for any not listed.
        **attrs
            Metadata for ``ds.attrs``: ``sample_name``, ``sample_weight_mg``,
            ``start_time`` (anything :class:`pandas.Timestamp` reads, stored
            as an ISO string), ``comment``, ``username``, ... ``None`` values
            are dropped.

        Returns
        -------
        TPRData
            New instance.
        """
        return cls._build(time_s, temperature_C, dict(signals), attrs, timestamp=timestamp, units=units)

    @classmethod
    @records_step
    def from_labview_tcd(cls, filepath: Union[str, Path]) -> "TPRData":
        """Read one export of the LabVIEW TPR rig (either column variant).

        The detector signals are stored in mV with the sign of the file;
        unrecognised columns are kept as further signals with ``units``
        ``'unknown'`` (the parser warns about them).

        Parameters
        ----------
        filepath : str or pathlib.Path
            The ``.txt`` export.

        Returns
        -------
        TPRData
            The run, its history starting with this step.
        """
        from phd_parser.tpr.labview_tcd import read_export

        raw = read_export(filepath)
        data, meta = raw["data"], raw["meta"]
        signals = dict(data["signals"])
        units = {name: SIGNAL_UNIT for name in signals}
        for name, values in data.get("extra", {}).items():
            signals[name] = values
            units[name] = "unknown"
        stored_to_original = {stored: original for original, stored in meta["column_map"].items()}
        source_columns = {TEMPERATURE: stored_to_original.get("temperature_C", "TC 1")}
        for name in signals:
            original = stored_to_original.get(name, meta["extra_columns"].get(name))
            if original is not None:
                source_columns[name] = str(original)
        attrs = {
            "sample_name": meta["sample_name"],
            "sample_weight_mg": meta["sample_weight_mg"],
            "start_time": meta["start_time"],
            "comment": meta["comment"],
            "username": meta["username"],
            "source_file": meta["source_file"],
            "format_variant": meta["format_variant"],
            "columns": list(meta["columns"]),
        }
        return cls._build(data["time_s"], data["temperature_C"], signals, attrs,
                          timestamp=data.get("timestamp"), units=units, source_columns=source_columns)

    @classmethod
    @records_step
    def from_netcdf(cls, filepath: Union[str, Path]) -> "TPRData":
        """Reload a run saved with :meth:`to_netcdf`.

        Parameters
        ----------
        filepath : str or pathlib.Path
            The ``.nc`` file.

        Returns
        -------
        TPRData
            The instance, its history extended by this load.
        """
        with xr.open_dataset(filepath, engine="h5netcdf") as handle:
            ds = handle.load()
        ds.attrs = decode_attrs(dict(ds.attrs))
        for name in ds.variables:
            ds[name].attrs = decode_attrs(dict(ds[name].attrs))
        return cls(ds=ds)

    def __repr__(self) -> str:
        """One line: sample, points, time span, temperature span, signals."""
        time = self.time
        temperature = self.temperature
        time_span = f"{time.min():.0f}-{time.max():.0f} s" if time.size else "empty"
        temperature_span = (f"{np.nanmin(temperature):.0f}-{np.nanmax(temperature):.0f} {TEMPERATURE_UNIT}"
                            if temperature.size and np.isfinite(temperature).any() else "empty")
        return (f"TPRData(sample_name={self.sample_name!r}, n_points={self.n_points}, time={time_span}, "
                f"temperature={temperature_span}, signals={self.signals})")
