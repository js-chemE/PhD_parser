"""Smoothing, averaging, normalisation, Gram-Schmidt and signal-to-noise estimators.

A mixin of :class:`~phd_parser.spectroscopy.core.SpectroscopyData`; never used on its own.
"""

import logging
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, Tuple, Union
from typing import Any, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import xarray as xr

from phd_parser.history import records_step

if TYPE_CHECKING:
    from phd_parser.spectroscopy.core import SpectroscopyData

logger = logging.getLogger(__name__)


class _SignalMixin:
    """Smoothing, averaging, normalisation, Gram-Schmidt and signal-to-noise estimators."""

    @records_step
    def smooth_savgol(self, window_length: int = 21, polyorder: int = 3) -> "SpectroscopyData":
        """Apply a Savitzky-Golay smoothing filter along the wavenumber axis.

        Parameters
        ----------
        window_length : int, optional
            Length of the filter window in points (default is ``21``).
        polyorder : int, optional
            Polynomial order for the filter (default is ``3``).

        Returns
        -------
        SpectroscopyData
            New instance with smoothed spectral values.
        """
        from scipy.signal import savgol_filter

        def _filter(arr: npt.NDArray) -> npt.NDArray:
            return savgol_filter(arr, window_length, polyorder, axis=-1)

        smoothed = _filter(self.values)
        smoothed_baseline = _filter(self.baseline) if self.has_baseline else None
        ds_new = self._build_ds(self.wavenumber, smoothed, tos=self.tos, attrs=dict(self.ds.attrs))
        ds_new = self._with_baseline(ds_new, smoothed_baseline)
        return self._new(ds=self._carry_auxiliary(ds_new))

    @records_step
    def smooth_moving(self, window_size: int = 5) -> "SpectroscopyData":
        """Apply a uniform moving-average filter along the wavenumber axis.

        Parameters
        ----------
        window_size : int, optional
            Number of points in the averaging window (default is ``5``).

        Returns
        -------
        SpectroscopyData
            New instance with smoothed spectral values.

        Raises
        ------
        ValueError
            If ``window_size`` is less than ``1``.
        """
        if window_size < 1:
            raise ValueError("window_size must be >= 1")

        kernel = np.ones(window_size) / window_size

        def _filter(arr: npt.NDArray) -> npt.NDArray:
            if arr.ndim == 1:
                return np.convolve(arr, kernel, mode="same")
            return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=-1, arr=arr)

        smoothed = _filter(self.values)
        smoothed_baseline = _filter(self.baseline) if self.has_baseline else None
        ds_new = self._build_ds(self.wavenumber, smoothed, tos=self.tos, attrs=dict(self.ds.attrs))
        ds_new = self._with_baseline(ds_new, smoothed_baseline)
        return self._new(ds=self._carry_auxiliary(ds_new))

    @records_step
    def smooth_gaussian(self, sigma_cm: float) -> "SpectroscopyData":
        """Apply a Gaussian smoothing filter along the wavenumber axis.

        The kernel width is given in cm⁻¹ and converted to points with the
        grid spacing of the wavenumber axis (its median, if the grid is not
        uniform, with a warning).

        Parameters
        ----------
        sigma_cm : float
            Standard deviation of the Gaussian kernel in cm⁻¹.

        Returns
        -------
        SpectroscopyData
            New instance with smoothed spectral values.

        Raises
        ------
        ValueError
            If ``sigma_cm`` is negative or the axis has fewer than 2 points.
        """
        from scipy.ndimage import gaussian_filter1d

        if sigma_cm < 0:
            raise ValueError(f"sigma_cm must be >= 0, got {sigma_cm}")
        spacing_cm = np.abs(np.diff(self.wavenumber_per_cm))
        if spacing_cm.size == 0:
            raise ValueError("smooth_gaussian needs at least 2 wavenumber points")
        step_cm = float(np.median(spacing_cm))
        if float(spacing_cm.max() - spacing_cm.min()) > 0.01 * step_cm:
            logger.warning(
                f"Wavenumber grid is not uniform ({spacing_cm.min():.4g}–{spacing_cm.max():.4g} cm⁻¹); "
                f"converting sigma with the median spacing {step_cm:.4g} cm⁻¹."
            )
        sigma_points = sigma_cm / step_cm

        def _filter(arr: npt.NDArray) -> npt.NDArray:
            return gaussian_filter1d(arr, sigma=sigma_points, axis=-1)

        smoothed = _filter(self.values)
        smoothed_baseline = _filter(self.baseline) if self.has_baseline else None
        ds_new = self._build_ds(self.wavenumber, smoothed, tos=self.tos, attrs=dict(self.ds.attrs))
        ds_new = self._with_baseline(ds_new, smoothed_baseline)
        return self._new(ds=self._carry_auxiliary(ds_new))

    @records_step
    def average_scans(
        self,
        number_of_scans: int,
        tos_method: Literal["mean", "median", "first", "last"] = "first",
    ) -> "SpectroscopyData":
        """Co-add consecutive groups of scans by averaging.

        Trailing scans that do not fill a complete group are discarded.

        Parameters
        ----------
        number_of_scans : int
            Number of consecutive scans per group.
        tos_method : {'mean', 'median', 'first', 'last'}, optional
            How to assign a representative ``tos`` value to each averaged
            group (default is ``'first'``).

        Returns
        -------
        SpectroscopyData
            New instance with ``n_scan // number_of_scans`` averaged scans.

        Raises
        ------
        ValueError
            If the data are 1-D or if ``number_of_scans`` is less than ``1``.
        """
        if self.ndim != 2:
            raise ValueError("average_scans requires 2-D data")
        if number_of_scans < 1:
            raise ValueError("number_of_scans must be >= 1")

        n_averaged = self.shape[0] // number_of_scans
        new_values = (
            self.values[: n_averaged * number_of_scans]
            .reshape(n_averaged, number_of_scans, -1)
            .mean(axis=1)
        )
        new_baseline = None
        if self.has_baseline:
            new_baseline = (
                self.baseline[: n_averaged * number_of_scans]
                .reshape(n_averaged, number_of_scans, -1)
                .mean(axis=1)
            )

        new_tos = None
        if self.tos is not None:
            tos_blocks = self.tos[: n_averaged * number_of_scans].reshape(n_averaged, number_of_scans)
            if tos_method == "mean":
                new_tos = tos_blocks.mean(axis=1)
            elif tos_method == "median":
                new_tos = np.median(tos_blocks, axis=1)
            elif tos_method == "first":
                new_tos = tos_blocks[:, 0]
            elif tos_method == "last":
                new_tos = tos_blocks[:, -1]

        # tos values remain absolute elapsed seconds, so tos_start stays valid
        ds_new = self._build_ds(self.wavenumber, new_values, tos=new_tos, attrs=dict(self.ds.attrs))
        ds_new = self._with_baseline(ds_new, new_baseline)
        return self._new(ds=self._carry_auxiliary(ds_new))

    @records_step
    def average_scans_by_tos(
        self,
        target_tos: Union[float, Sequence[float]],
        method: Literal["nearest", "pad", "backfill"] = "nearest",
        tolerance_seconds: Optional[float] = 10,
        number_of_scans: Optional[int] = None,
        time_window: Optional[float] = None,
        direction: Literal["forward", "backward", "center"] = "center",
    ) -> "SpectroscopyData":
        """Return a new SpectroscopyData from scans averaged around each target tos.

        Each target produces one averaged spectrum.  The anchor ``tos`` of the
        nearest real scan is used as the ``tos`` coordinate for that output
        spectrum.

        Parameters
        ----------
        target_tos : float or sequence of float
            Target elapsed time(s) in seconds.
        method : {'nearest', 'pad', 'backfill'}, optional
            Selection method for the anchor scan (default is ``'nearest'``).
        tolerance_seconds : float or None, optional
            Maximum distance in seconds between a target and its anchor scan
            (default is ``10``).
        number_of_scans : int or None, optional
            Number of scans to average.  Mutually exclusive with
            ``time_window``.
        time_window : float or None, optional
            Duration in seconds over which scans are averaged.  Mutually
            exclusive with ``number_of_scans``.
        direction : {'forward', 'backward', 'center'}, optional
            Position of the anchor within the window (default is
            ``'center'``).

        Returns
        -------
        SpectroscopyData
            New instance with one scan per target tos.

        Raises
        ------
        ValueError
            If the data are 1-D or if any target exceeds the tolerance.
        """
        if self.ndim != 2:
            raise ValueError("average_scans_by_tos requires 2-D data")

        scalar_input = np.ndim(target_tos) == 0
        targets = [float(target_tos)] if scalar_input else [float(t) for t in target_tos]
        targets = sorted(targets)  # the result is a series: stored in time order, with its baseline

        # Reuse the averaging logic: returns list of 1-D arrays (or one array if scalar)
        averaged = self.get_scan_by_tos_average(
            target_tos=targets[0] if scalar_input else targets,
            method=method,
            tolerance_seconds=tolerance_seconds,
            number_of_scans=number_of_scans,
            time_window=time_window,
            direction=direction,
        )

        # Normalise to list of 1-D arrays regardless of scalar/array input
        if scalar_input:
            averaged_list = [averaged]
        else:
            averaged_list = averaged  # already a list

        new_values = np.vstack(averaged_list)  # (n_targets, n_wavenumber)

        new_baseline = None
        if self.has_baseline:
            baseline_ir = self._new(ds=self._build_ds(self.wavenumber, self.baseline, tos=self.tos, attrs={}))
            bl_averaged = baseline_ir.get_scan_by_tos_average(
                target_tos=targets[0] if scalar_input else targets,
                method=method,
                tolerance_seconds=tolerance_seconds,
                number_of_scans=number_of_scans,
                time_window=time_window,
                direction=direction,
            )
            bl_list = [bl_averaged] if scalar_input else bl_averaged
            new_baseline = np.vstack(bl_list)

        # Anchor tos: the nearest actual tos to each target becomes the new coord
        tos_values = self.tos
        anchor_tos = [float(tos_values[int(np.abs(tos_values - t).argmin())]) for t in targets]
        new_tos = np.array(anchor_tos)

        new_attrs = {
            **self.ds.attrs,
            "averaged_target_tos": [float(t) for t in targets],
            "averaged_anchor_tos": anchor_tos,
            "averaged_direction": direction,
            "averaged_number_of_scans": number_of_scans,
            "averaged_time_window": time_window,
        }

        ds_new = self._build_ds(
            wavenumber_si=self.wavenumber,
            values=new_values,
            tos=new_tos,
            attrs=new_attrs,
        )
        ds_new = self._with_baseline(ds_new, new_baseline)
        return self._new(ds=self._carry_auxiliary(ds_new))

    @records_step
    def normalise_max(self) -> "SpectroscopyData":
        """Divide all values by the global maximum.

        Returns
        -------
        SpectroscopyData
            New instance with values in ``[0, 1]``.  Returns ``self``
            unchanged if the maximum is zero.
        """
        max_val = self.values.max()
        if max_val == 0:
            logger.warning("Maximum value is zero; returning original data without normalisation")
            return self
        new_values = self.values / max_val
        new_baseline = self.baseline / max_val if self.has_baseline else None
        ds_new = self._build_ds(self.wavenumber, new_values, tos=self.tos, attrs=dict(self.ds.attrs))
        ds_new = self._with_baseline(ds_new, new_baseline)
        return self._new(ds=self._carry_auxiliary(ds_new))

    @records_step
    def normalise_integral(self) -> "SpectroscopyData":
        """Divide each spectrum by its trapezoidal integral over the wavenumber axis.

        Returns
        -------
        SpectroscopyData
            New instance with unit-integral spectra.  Returns ``self``
            unchanged if any integral is zero.
        """
        integral = np.trapezoid(self.values, x=self.wavenumber, axis=-1)
        if np.any(integral == 0):
            logger.warning("Integral is zero for some scans; returning original data without normalisation")
            return self
        new_values = self.values / integral[..., np.newaxis]
        new_baseline = self.baseline / integral[..., np.newaxis] if self.has_baseline else None
        ds_new = self._build_ds(self.wavenumber, new_values, tos=self.tos, attrs=dict(self.ds.attrs))
        ds_new = self._with_baseline(ds_new, new_baseline)
        return self._new(ds=self._carry_auxiliary(ds_new))

    @records_step
    def normalise_reference(self, reference: npt.NDArray) -> "SpectroscopyData":
        """Divide all spectra element-wise by a reference spectrum.

        Parameters
        ----------
        reference : numpy.ndarray
            1-D reference spectrum with the same number of wavenumber points.

        Returns
        -------
        SpectroscopyData
            New instance with values divided by ``reference``.  Returns
            ``self`` unchanged if ``reference`` contains any zeros.

        Raises
        ------
        ValueError
            If ``reference`` is not 1-D or its size does not match the
            wavenumber axis.
        """
        if reference.ndim != 1:
            raise ValueError("Reference spectrum must be 1-D")
        if reference.size != self.wavenumber.size:
            raise ValueError(f"Reference size ({reference.size}) does not match wavenumber size ({self.wavenumber.size})")
        if np.any(reference == 0):
            logger.warning("Reference spectrum contains zero values; returning original data without normalisation")
            return self

        new_values = self.values / reference
        new_baseline = self.baseline / reference if self.has_baseline else None
        ds_new = self._build_ds(self.wavenumber, new_values, tos=self.tos, attrs=dict(self.ds.attrs))
        ds_new = self._with_baseline(ds_new, new_baseline)
        return self._new(ds=self._carry_auxiliary(ds_new))

    @records_step
    def normalise_reference_scan(self, scan_index: int) -> "SpectroscopyData":
        """Normalise all spectra by the scan at ``scan_index``.

        Parameters
        ----------
        scan_index : int
            Zero-based index of the scan to use as the reference.

        Returns
        -------
        SpectroscopyData
            New instance with each spectrum divided by the reference scan.

        Raises
        ------
        ValueError
            If the data are 1-D.
        """
        if self.ndim != 2:
            raise ValueError("normalise_reference_scan requires 2-D data")
        reference = self.get_scan(scan_index)
        return self.normalise_reference(reference)

    @records_step
    def normalise_reference_by_tos(
        self,
        target_tos: float,
        method: Literal["nearest", "pad", "backfill"] = "nearest",
        tolerance_seconds: Optional[float] = 10,
    ) -> "SpectroscopyData":
        """Normalise all spectra by the scan nearest to ``target_tos``.

        Parameters
        ----------
        target_tos : float
            Target elapsed time in seconds.
        method : {'nearest', 'pad', 'backfill'}, optional
            Selection method for the reference scan (default is
            ``'nearest'``).
        tolerance_seconds : float or None, optional
            Maximum distance in seconds between the target and the nearest
            scan (default is ``10``).

        Returns
        -------
        SpectroscopyData
            New instance with each spectrum divided by the reference scan.

        Raises
        ------
        ValueError
            If the data are 1-D or if the target exceeds the tolerance.
        """
        if self.ndim != 2:
            raise ValueError("normalise_reference_by_tos requires 2-D data")
        reference = self.get_scan_by_tos(target_tos, method=method, tolerance_seconds=tolerance_seconds)
        return self.normalise_reference(reference)

    @records_step
    def normalise_value_range(self, new_min: float = 0.0, new_max: float = 1.0) -> "SpectroscopyData":
        """Rescale values to fit within ``[new_min, new_max]``.

        Parameters
        ----------
        new_min : float, optional
            Target minimum value (default is ``0.0``).
        new_max : float, optional
            Target maximum value (default is ``1.0``).

        Returns
        -------
        SpectroscopyData
            New instance with linearly rescaled values.  Returns ``self``
            unchanged if all values are identical.
        """
        old_min = self.values.min()
        old_max = self.values.max()
        if old_max == old_min:
            logger.warning("All values are the same; returning original data without normalisation")
            return self
        scale = (new_max - new_min) / (old_max - old_min)
        new_values = (self.values - old_min) * scale + new_min
        # Baseline is scaled (not shifted) so that data + baseline transforms consistently:
        # the shift applies once to data, while the additive baseline only needs rescaling.
        new_baseline = self.baseline * scale if self.has_baseline else None
        ds_new = self._build_ds(self.wavenumber, new_values, tos=self.tos, attrs=dict(self.ds.attrs))
        ds_new = self._with_baseline(ds_new, new_baseline)
        return self._new(ds=self._carry_auxiliary(ds_new))

    @records_step
    def normalise_value(self, factor: float) -> "SpectroscopyData":
        """Divide all values by a scalar factor.

        Parameters
        ----------
        factor : float
            The divisor.  Must be non-zero.

        Returns
        -------
        SpectroscopyData
            New instance with values divided by ``factor``.  Returns ``self``
            unchanged if ``factor`` is zero.
        """
        if factor == 0:
            logger.warning("Normalisation factor is zero; returning original data without normalisation")
            return self
        new_values = self.values / factor
        new_baseline = self.baseline / factor if self.has_baseline else None
        ds_new = self._build_ds(self.wavenumber, new_values, tos=self.tos, attrs=dict(self.ds.attrs))
        ds_new = self._with_baseline(ds_new, new_baseline)
        return self._new(ds=self._carry_auxiliary(ds_new))

    def get_gram_schmidt(self, reference: npt.NDArray) -> xr.DataArray:
        """Gram-Schmidt vector: L2 norm of each scan's component orthogonal to the reference subspace.

        Parameters
        ----------
        reference : numpy.ndarray
            1-D or 2-D array of reference spectrum/spectra with shape
            ``(n_wavenumber,)`` or ``(n_ref, n_wavenumber)``.

        Returns
        -------
        xr.DataArray
            1-D DataArray of shape ``(n_scan,)`` containing the Gram-Schmidt
            orthogonal norm for each scan.  Includes ``'tos'`` as a
            non-dimension coordinate when available.

        Raises
        ------
        ValueError
            If the data are 1-D, if the reference wavenumber dimension does
            not match, or if the reference spectra are linearly dependent.
        """
        if self.ndim != 2:
            raise ValueError("get_gram_schmidt requires 2-D data")

        ref = np.atleast_2d(np.asarray(reference, dtype=float))
        if ref.shape[-1] != self.wavenumber.size:
            raise ValueError(
                f"Reference has {ref.shape[-1]} points but wavenumber axis has {self.wavenumber.size}"
            )

        basis: list[npt.NDArray] = []
        for r in ref:
            v = r.copy()
            for b in basis:
                v = v - np.dot(v, b) * b
            norm = np.linalg.norm(v)
            ref_norm = np.linalg.norm(r)
            if norm > 1e-12 * (ref_norm if ref_norm > 0 else 1.0):
                basis.append(v / norm)

        if not basis:
            raise ValueError("Reference spectra are linearly dependent; cannot build orthonormal basis")

        B = np.array(basis)                          # (n_basis, n_wn)
        spectra = self.values                        # (n_scans, n_wn)
        residuals = spectra - (spectra @ B.T) @ B   # remove reference subspace
        gs_values = np.linalg.norm(residuals, axis=1)  # (n_scans,)

        coords: dict[str, Any] = {"scan": self.ds.coords["scan"].values}
        if self.tos is not None:
            coords["tos"] = ("scan", self.tos)

        return xr.DataArray(
            data=gs_values,
            coords=coords,
            dims=["scan"],
            attrs={"gs_n_basis": len(basis)},
            name="gram_schmidt",
        )

    def get_gram_schmidt_scan(
        self,
        reference_scans: Union[int, Sequence[int]] = 0,
    ) -> xr.DataArray:
        """Compute the Gram-Schmidt chromatogram using one or more scans as the reference.

        Parameters
        ----------
        reference_scans : int or sequence of int, optional
            Zero-based index or indices of the scan(s) to use as the reference
            subspace (default is ``0``).

        Returns
        -------
        xr.DataArray
            1-D DataArray of Gram-Schmidt norms, one per scan.

        Raises
        ------
        ValueError
            If the data are 1-D.
        """
        if self.ndim != 2:
            raise ValueError("get_gram_schmidt_scan requires 2-D data")
        indices = [int(reference_scans)] if np.ndim(reference_scans) == 0 else [int(i) for i in reference_scans]
        return self.get_gram_schmidt(self.values[indices])

    def get_gram_schmidt_by_tos(
        self,
        reference_tos: Union[float, Sequence[float]],
        method: Literal["nearest", "pad", "backfill"] = "nearest",
        tolerance_seconds: Optional[float] = 10,
    ) -> xr.DataArray:
        """Compute the Gram-Schmidt chromatogram using scans selected by tos as the reference.

        Parameters
        ----------
        reference_tos : float or sequence of float
            Target elapsed time(s) in seconds identifying the reference
            scan(s).
        method : {'nearest', 'pad', 'backfill'}, optional
            Selection method (default is ``'nearest'``).
        tolerance_seconds : float or None, optional
            Maximum distance in seconds from the target to the nearest scan
            (default is ``10``).

        Returns
        -------
        xr.DataArray
            1-D DataArray of Gram-Schmidt norms, one per scan.

        Raises
        ------
        ValueError
            If the data are 1-D or if any target exceeds the tolerance.
        """
        if self.ndim != 2:
            raise ValueError("get_gram_schmidt_by_tos requires 2-D data")
        reference = self.get_scan_by_tos(
            reference_tos, method=method, tolerance_seconds=tolerance_seconds
        )
        return self.get_gram_schmidt(reference)

    def _snr_apply(self, fn) -> Union[float, npt.NDArray]:
        if self.ndim == 1:
            return float(fn(self.values))
        rows = self.values.reshape(-1, self.shape[-1])
        return np.array([fn(row) for row in rows]).reshape(self.shape[:-1])

    def snr_windows(
        self,
        signal_range_cm: Tuple[float, float],
        noise_range_cm: Tuple[float, float],
        signal_metric: Literal["max", "peak_to_peak", "integral", "rms"] = "max",
        noise_metric: Literal["std", "rms", "peak_to_peak"] = "std",
    ) -> Union[float, npt.NDArray]:
        """Estimate SNR using separate spectral windows for signal and noise.

        Parameters
        ----------
        signal_range_cm : tuple of float
            ``(low, high)`` wavenumber range in cm⁻¹ defining the signal
            window.
        noise_range_cm : tuple of float
            ``(low, high)`` wavenumber range in cm⁻¹ defining the noise
            window.
        signal_metric : {'max', 'peak_to_peak', 'integral', 'rms'}, optional
            Statistic used to quantify the signal (default is ``'max'``).
        noise_metric : {'std', 'rms', 'peak_to_peak'}, optional
            Statistic used to quantify the noise (default is ``'std'``).

        Returns
        -------
        float or numpy.ndarray
            Scalar SNR for 1-D data; array of shape ``(n_scan,)`` for 2-D
            data.

        Raises
        ------
        ValueError
            If no wavenumber points fall within either the signal or noise
            range.
        """
        wn_si = self.wavenumber
        sig_lo, sig_hi = sorted(signal_range_cm)
        noi_lo, noi_hi = sorted(noise_range_cm)
        sig_mask = (wn_si >= sig_lo * 100.0) & (wn_si <= sig_hi * 100.0)
        noi_mask = (wn_si >= noi_lo * 100.0) & (wn_si <= noi_hi * 100.0)
        if not sig_mask.any():
            raise ValueError(f"No points in signal range {sig_lo:.0f}–{sig_hi:.0f} cm⁻¹")
        if not noi_mask.any():
            raise ValueError(f"No points in noise range {noi_lo:.0f}–{noi_hi:.0f} cm⁻¹")

        wn_sig = wn_si[sig_mask]

        def _signal(spec_1d: np.ndarray) -> float:
            s = spec_1d[sig_mask]
            if signal_metric == "max":
                return float(np.abs(s).max())
            if signal_metric == "peak_to_peak":
                return float(s.max() - s.min())
            if signal_metric == "integral":
                return float(np.abs(np.trapezoid(s, x=wn_sig)))
            return float(np.sqrt(np.mean(s ** 2)))  # rms

        def _noise(spec_1d: np.ndarray) -> float:
            n = spec_1d[noi_mask]
            if noise_metric == "std":
                return float(n.std(ddof=1))
            if noise_metric == "rms":
                return float(np.sqrt(np.mean(n ** 2)))
            return float(n.max() - n.min())  # peak_to_peak

        def _ratio(spec_1d: np.ndarray) -> float:
            sigma = _noise(spec_1d)
            if sigma == 0:
                logger.warning("Noise is zero; returning inf")
                return float("inf")
            return _signal(spec_1d) / sigma

        return self._snr_apply(_ratio)

    def snr_noise_window(
        self,
        noise_range_cm: Tuple[float, float],
        signal_range_cm: Optional[Tuple[float, float]] = None,
        noise_metric: Literal["rms", "std", "peak_to_peak"] = "rms",
        detrend_order: int = 1,
    ) -> Union[float, npt.NDArray]:
        """Estimate SNR using a dedicated noise window with polynomial detrending.

        The noise region is optionally detrended before computing the noise
        metric.  The signal is taken as the absolute maximum over
        ``signal_range_cm`` (or the entire spectrum if ``None``).

        Parameters
        ----------
        noise_range_cm : tuple of float
            ``(low, high)`` wavenumber range in cm⁻¹ used to estimate noise.
        signal_range_cm : tuple of float or None, optional
            ``(low, high)`` wavenumber range in cm⁻¹ used to find the peak
            signal.  ``None`` uses the full spectrum (default is ``None``).
        noise_metric : {'rms', 'std', 'peak_to_peak'}, optional
            Statistic used to quantify the detrended noise (default is
            ``'rms'``).
        detrend_order : int, optional
            Polynomial order for detrending the noise window.  Use ``-1`` to
            skip detrending (default is ``1``).

        Returns
        -------
        float or numpy.ndarray
            Scalar SNR for 1-D data; array of shape ``(n_scan,)`` for 2-D
            data.

        Raises
        ------
        ValueError
            If no points fall within the noise or signal range.
        """
        wn_si = self.wavenumber
        noi_lo, noi_hi = sorted(noise_range_cm)
        noi_mask = (wn_si >= noi_lo * 100.0) & (wn_si <= noi_hi * 100.0)
        if not noi_mask.any():
            raise ValueError(f"No points in noise range {noi_lo:.0f}–{noi_hi:.0f} cm⁻¹")

        if signal_range_cm is not None:
            sig_lo, sig_hi = sorted(signal_range_cm)
            sig_mask = (wn_si >= sig_lo * 100.0) & (wn_si <= sig_hi * 100.0)
            if not sig_mask.any():
                raise ValueError(f"No points in signal range {sig_lo:.0f}–{sig_hi:.0f} cm⁻¹")
        else:
            sig_mask = np.ones_like(wn_si, dtype=bool)

        wn_noi = wn_si[noi_mask]

        def _noise(spec_1d: np.ndarray) -> float:
            n = spec_1d[noi_mask].astype(float)
            if detrend_order >= 0:
                coeffs = np.polyfit(wn_noi, n, deg=detrend_order)
                n = n - np.polyval(coeffs, wn_noi)
            if noise_metric == "rms":
                return float(np.sqrt(np.mean(n ** 2)))
            if noise_metric == "std":
                return float(n.std(ddof=1))
            return float(n.max() - n.min())  # peak_to_peak

        def _ratio(spec_1d: np.ndarray) -> float:
            sigma = _noise(spec_1d)
            if sigma == 0:
                logger.warning("Noise is zero; returning inf")
                return float("inf")
            return float(np.abs(spec_1d[sig_mask]).max()) / sigma

        return self._snr_apply(_ratio)

    def snr_der(
        self,
        signal_range_cm: Optional[Tuple[float, float]] = None,
    ) -> Union[float, npt.NDArray]:
        """Estimate SNR using the DER-SNR algorithm (Stoehr et al. 2008).

        The noise standard deviation is estimated from the second-difference
        of the spectrum without requiring a dedicated noise window.

        Parameters
        ----------
        signal_range_cm : tuple of float or None, optional
            ``(low, high)`` wavenumber range in cm⁻¹ used to find the peak
            signal.  ``None`` uses the full spectrum (default is ``None``).

        Returns
        -------
        float or numpy.ndarray
            Scalar SNR for 1-D data; array of shape ``(n_scan,)`` for 2-D
            data.

        Raises
        ------
        ValueError
            If no points fall within ``signal_range_cm`` or if the spectrum
            has fewer than 5 points.
        """
        wn_si = self.wavenumber
        if signal_range_cm is not None:
            sig_lo, sig_hi = sorted(signal_range_cm)
            sig_mask = (wn_si >= sig_lo * 100.0) & (wn_si <= sig_hi * 100.0)
            if not sig_mask.any():
                raise ValueError(f"No points in signal range {sig_lo:.0f}–{sig_hi:.0f} cm⁻¹")
        else:
            sig_mask = np.ones_like(wn_si, dtype=bool)

        factor = 1.482602 / np.sqrt(6.0)

        def _ratio(spec_1d: np.ndarray) -> float:
            if spec_1d.size < 5:
                raise ValueError("DER-SNR requires at least 5 points")
            diff = 2.0 * spec_1d[2:-2] - spec_1d[:-4] - spec_1d[4:]
            sigma = factor * float(np.median(np.abs(diff)))
            if sigma == 0:
                logger.warning("Noise is zero; returning inf")
                return float("inf")
            return float(np.abs(spec_1d[sig_mask]).max()) / sigma

        return self._snr_apply(_ratio)

    def snr_repeat(
        self,
        signal_range_cm: Optional[Tuple[float, float]] = None,
        reduce: Literal["max", "median", "mean", "per_wavenumber"] = "max",
    ) -> Union[float, npt.NDArray]:
        """Estimate SNR from scan-to-scan reproducibility across repeated measurements.

        The noise is the standard deviation across scans; the signal is the
        mean absolute value.

        Parameters
        ----------
        signal_range_cm : tuple of float or None, optional
            ``(low, high)`` wavenumber range in cm⁻¹ used to find the peak
            SNR.  ``None`` uses the full spectrum (default is ``None``).
        reduce : {'max', 'median', 'mean', 'per_wavenumber'}, optional
            How to summarise the per-wavenumber SNR across the signal range
            (default is ``'max'``).  Use ``'per_wavenumber'`` to return the
            full SNR spectrum.

        Returns
        -------
        float or numpy.ndarray
            Scalar SNR when ``reduce`` is not ``'per_wavenumber'``; array of
            shape ``(n_wavenumber,)`` when ``reduce='per_wavenumber'``.

        Raises
        ------
        ValueError
            If the data are 1-D, if fewer than 2 scans are present, or if no
            points fall within ``signal_range_cm``.
        """
        if self.ndim != 2:
            raise ValueError("snr_repeat requires 2-D data")
        if self.shape[0] < 2:
            raise ValueError("snr_repeat requires at least 2 scans")

        wn_si = self.wavenumber
        if signal_range_cm is not None:
            sig_lo, sig_hi = sorted(signal_range_cm)
            sig_mask = (wn_si >= sig_lo * 100.0) & (wn_si <= sig_hi * 100.0)
            if not sig_mask.any():
                raise ValueError(f"No points in signal range {sig_lo:.0f}–{sig_hi:.0f} cm⁻¹")
        else:
            sig_mask = np.ones_like(wn_si, dtype=bool)

        mean_spec = self.values.mean(axis=0)
        sigma_spec = self.values.std(axis=0, ddof=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            snr_per_wn = np.where(sigma_spec > 0, np.abs(mean_spec) / sigma_spec, np.inf)

        if reduce == "per_wavenumber":
            return snr_per_wn

        snr_in_range = snr_per_wn[sig_mask]
        if reduce == "max":
            return float(np.nanmax(snr_in_range))
        if reduce == "median":
            return float(np.nanmedian(snr_in_range))
        return float(np.nanmean(snr_in_range))  # mean

    def snr_psd(
        self,
        signal_range_cm: Optional[Tuple[float, float]] = None,
        noise_fraction: float = 0.25,
        detrend_order: int = 1,
    ) -> Union[float, npt.NDArray]:
        """Estimate SNR from the high-frequency tail of the power spectral density.

        The noise power is estimated from the top ``noise_fraction`` of FFT
        frequency bins.

        Parameters
        ----------
        signal_range_cm : tuple of float or None, optional
            ``(low, high)`` wavenumber range in cm⁻¹ used to find the peak
            signal.  ``None`` uses the full spectrum (default is ``None``).
        noise_fraction : float, optional
            Fraction of high-frequency FFT bins treated as noise, in
            ``(0, 1)`` (default is ``0.25``).
        detrend_order : int, optional
            Polynomial order for detrending before the FFT.  Use ``-1`` to
            skip (default is ``1``).

        Returns
        -------
        float or numpy.ndarray
            Scalar SNR for 1-D data; array of shape ``(n_scan,)`` for 2-D
            data.

        Raises
        ------
        ValueError
            If ``noise_fraction`` is not in ``(0, 1)``, if the spectrum has
            fewer than 8 points, or if no points fall within
            ``signal_range_cm``.
        """
        if not (0.0 < noise_fraction < 1.0):
            raise ValueError("noise_fraction must be in (0, 1)")

        wn_si = self.wavenumber
        n_pts = wn_si.size
        if n_pts < 8:
            raise ValueError("snr_psd requires at least 8 points")

        if signal_range_cm is not None:
            sig_lo, sig_hi = sorted(signal_range_cm)
            sig_mask = (wn_si >= sig_lo * 100.0) & (wn_si <= sig_hi * 100.0)
            if not sig_mask.any():
                raise ValueError(f"No points in signal range {sig_lo:.0f}–{sig_hi:.0f} cm⁻¹")
        else:
            sig_mask = np.ones_like(wn_si, dtype=bool)

        # Number of high-frequency FFT bins to treat as noise
        n_freq = n_pts // 2 + 1
        n_noise_bins = max(1, int(np.floor(n_freq * noise_fraction)))

        x_axis = np.arange(n_pts, dtype=float)

        def _noise(spec_1d: np.ndarray) -> float:
            s = spec_1d.astype(float)
            if detrend_order >= 0:
                coeffs = np.polyfit(x_axis, s, deg=detrend_order)
                s = s - np.polyval(coeffs, x_axis)
            spectrum = np.fft.rfft(s)
            psd = (np.abs(spectrum) ** 2) / n_pts
            tail = psd[-n_noise_bins:]
            # Parseval: noise variance ≈ mean PSD over the tail bins
            return float(np.sqrt(tail.mean()))

        def _ratio(spec_1d: np.ndarray) -> float:
            sigma = _noise(spec_1d)
            if sigma == 0:
                logger.warning("Noise is zero; returning inf")
                return float("inf")
            return float(np.abs(spec_1d[sig_mask]).max()) / sigma

        return self._snr_apply(_ratio)

    def find_spike_scans(
        self,
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

        if self.ndim != 2:
            raise ValueError("find_spike_scans requires 2-D data")
        if window < 3 or window % 2 == 0:
            raise ValueError("window must be an odd integer >= 3")
        values = self.values[:, np.isfinite(self.values).all(axis=0)]
        running_median = median_filter(values, size=(window, 1), mode="nearest")
        deviation = np.abs(values - running_median)
        second_difference = np.diff(values, n=2, axis=0)
        noise = 1.4826 * np.median(np.abs(second_difference), axis=0) / np.sqrt(6.0)
        flagged = deviation > threshold * np.where(noise > 0, noise, np.inf)
        return flagged.mean(axis=1) >= min_fraction
