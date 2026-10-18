"""The single-beam background and the ``data_type`` conversions that depend on it.

A mixin of :class:`~phd_parser.spectroscopy.core.SpectroscopyData`; never used on its own.
"""

import logging
from typing import TYPE_CHECKING
from typing import Literal, Optional, Union

import numpy as np
import numpy.typing as npt
import xarray as xr

from phd_parser.history import records_step
from phd_parser.spectroscopy._common import (
    DataType,
    _as_single_spectrum,
    _is_spectroscopy_data,
)

if TYPE_CHECKING:
    from phd_parser.spectroscopy.core import SpectroscopyData

logger = logging.getLogger(__name__)


class _BackgroundMixin:
    """The single-beam background and the ``data_type`` conversions that depend on it."""

    @records_step
    def with_background(
        self,
        background: Union[npt.NDArray, "SpectroscopyData"],
        data_type: Optional[DataType] = None,
    ) -> "SpectroscopyData":
        """Set or switch the background spectrum (always stored as single_beam).

        If no background is currently assigned: assigns it without touching data values.
        If a background is already assigned: converts the new background to single_beam
        using the existing background, then recalculates data values accordingly.

        Parameters
        ----------
        background : numpy.ndarray or SpectroscopyData
            Background spectrum.  If an ``SpectroscopyData`` its ``data_type`` is used
            unless overridden by ``data_type``; a one-scan 2-D instance (what
            reading a single ``.spa`` file gives) counts as a single spectrum.
        data_type : str or None, optional
            Override the data type of ``background`` when it is a plain array.
            Must be one of the ``DataType`` literals (default is ``None``).

        Returns
        -------
        SpectroscopyData
            New instance with the background set and data values recalculated
            if a previous background existed.

        Raises
        ------
        ValueError
            If the background ``SpectroscopyData`` holds more than one scan, if the
            background array is not 1-D, or if the background size does not
            match the wavenumber axis.
        """
        if _is_spectroscopy_data(background):
            bg_values = _as_single_spectrum(background)
            bg_data_type = data_type if data_type is not None else background.data_type
        else:
            bg_values = np.asarray(background, dtype=float)
            if bg_values.ndim != 1:
                raise ValueError("Background array must be 1-D")
            bg_data_type = data_type

        if bg_values.size != self.wavenumber.size:
            raise ValueError(
                f"Background size ({bg_values.size}) does not match wavenumber axis ({self.wavenumber.size})"
            )

        bg_sb = self._bg_to_single_beam(bg_values, bg_data_type)

        if not self.has_background:
            return self._set_background(bg_sb)
        else:
            return self._switch_background(bg_sb)

    @records_step
    def with_background_scan(self, scan_index: int) -> "SpectroscopyData":
        """Use the scan at scan_index as the new background.

        The scan is extracted as a 1-D SpectroscopyData with the same data_type as self.
        If data is not single_beam, the scan is converted back to single_beam using
        the existing background before being assigned.

        Parameters
        ----------
        scan_index : int
            Zero-based index of the scan to promote to background.

        Returns
        -------
        SpectroscopyData
            New instance with the selected scan set as background.

        Raises
        ------
        ValueError
            If the data are 1-D.
        """
        if self.ndim != 2:
            raise ValueError("with_background_scan requires 2-D data")
        return self.with_background(self.select_by_idx(scan_index))

    @records_step
    def with_background_by_tos(
        self,
        target_tos: float,
        method: Literal["nearest", "pad", "backfill"] = "nearest",
        tolerance_seconds: Optional[float] = 10,
    ) -> "SpectroscopyData":
        """Use the scan nearest to target_tos as the new background.

        Same single_beam conversion logic as with_background_scan.

        Parameters
        ----------
        target_tos : float
            Target elapsed time in seconds.
        method : {'nearest', 'pad', 'backfill'}, optional
            Selection method (default is ``'nearest'``).
        tolerance_seconds : float or None, optional
            Maximum distance in seconds from the target to the nearest scan
            (default is ``10``).

        Returns
        -------
        SpectroscopyData
            New instance with the selected scan set as background.

        Raises
        ------
        ValueError
            If the data are 1-D or if the target exceeds the tolerance.
        """
        if self.ndim != 2:
            raise ValueError("with_background_by_tos requires 2-D data")
        return self.with_background(self.select_by_tos(target_tos, method=method, tolerance_seconds=tolerance_seconds))

    @records_step
    def del_background(self) -> "SpectroscopyData":
        """Remove the background spectrum without changing data values.

        Returns
        -------
        SpectroscopyData
            New instance with the ``'background'`` variable dropped.
        """
        return self._del_background()

    @records_step
    def set_background(
        self,
        background: Union[npt.NDArray, "SpectroscopyData"],
        data_type: Optional[DataType] = None,
    ) -> "SpectroscopyData":
        """Force-assign a background without recalculating values (drops any existing background first).

        Use with_background() instead when switching backgrounds should trigger recalculation.

        Parameters
        ----------
        background : numpy.ndarray or SpectroscopyData
            Background spectrum to store.  If an ``SpectroscopyData`` its ``data_type``
            is used unless overridden by ``data_type``; a one-scan 2-D
            instance counts as a single spectrum.
        data_type : str or None, optional
            Override the data type of ``background`` when it is a plain array
            (default is ``None``).

        Returns
        -------
        SpectroscopyData
            New instance with the background assigned and data values
            unchanged.

        Raises
        ------
        ValueError
            If the background ``SpectroscopyData`` holds more than one scan, if the
            array is not 1-D, or if the background size does not match the
            wavenumber axis.
        """
        if _is_spectroscopy_data(background):
            bg_values = _as_single_spectrum(background)
            bg_data_type = data_type if data_type is not None else background.data_type
        else:
            bg_values = np.asarray(background, dtype=float)
            if bg_values.ndim != 1:
                raise ValueError("Background array must be 1-D")
            bg_data_type = data_type

        if bg_values.size != self.wavenumber.size:
            raise ValueError(
                f"Background size ({bg_values.size}) does not match wavenumber axis ({self.wavenumber.size})"
            )

        bg_sb = self._bg_to_single_beam(bg_values, bg_data_type)
        return self._del_background()._set_background(bg_sb)

    def _convert(self, new_values: npt.NDArray, new_data_type: DataType) -> "SpectroscopyData":
        self._warn_drop_baseline(
            f"converting data_type to '{new_data_type}' invalidates the baseline curve"
        )
        new_attrs = {**self.ds.attrs, "data_type": new_data_type}
        ds_new = self._build_ds(self.wavenumber, new_values, tos=self.tos, attrs=new_attrs)
        return self._new(ds=self._carry_auxiliary(ds_new))

    def _require_background(self) -> npt.NDArray:
        if not self.has_background:
            raise ValueError(
                f"Converting from '{self.data_type}' requires a background spectrum. "
                "Set one with .with_background()."
            )
        return self.background

    @records_step
    def to_single_beam(self) -> "SpectroscopyData":
        """Convert back to raw single-beam units by re-applying the stored background.

        The inverse of the other ``to_*`` conversions.  Needed before
        :meth:`merge`, which is only defined on single-beam data.

        Returns
        -------
        SpectroscopyData
            New instance with ``data_type='single_beam'``.  Returns ``self``
            if already single_beam.

        Raises
        ------
        ValueError
            If no background is stored or if ``data_type`` is unknown.
        """
        if self.data_type == "single_beam":
            return self
        bg = self._require_background()
        if self.data_type in ("transmittance", "reflectance"):
            return self._convert(self.values * bg, "single_beam")
        if self.data_type in ("absorbance", "log_1_r"):
            return self._convert(np.power(10.0, -self.values) * bg, "single_beam")
        if self.data_type == "kubelka_munk":
            km = self.values
            r = (1.0 + km) - np.sqrt(km * (km + 2.0))
            return self._convert(r * bg, "single_beam")
        raise ValueError(f"Cannot convert '{self.data_type}' to single_beam")

    @records_step
    def to_transmittance(self) -> "SpectroscopyData":
        """Convert to transmittance (T = sample / background or T = 10^−A).

        Returns
        -------
        SpectroscopyData
            New instance with ``data_type='transmittance'``.  Returns ``self``
            if already transmittance.

        Raises
        ------
        ValueError
            If ``data_type`` is a reflectance-based type (``'reflectance'``,
            ``'log_1_r'``, ``'kubelka_munk'``) or if a background is needed
            but not set.
        """
        if self.data_type == "transmittance":
            return self
        if self.data_type == "single_beam":
            bg = self._require_background()
            if np.any(bg == 0):
                logger.warning("Background contains zeros; transmittance will contain inf")
            return self._convert(self.values / bg, "transmittance")
        if self.data_type == "absorbance":
            return self._convert(np.power(10.0, -self.values), "transmittance")
        raise ValueError(
            f"Cannot convert '{self.data_type}' to transmittance. "
            "Reflectance-based types (reflectance, log_1_r, kubelka_munk) belong to a different experiment."
        )

    @records_step
    def to_reflectance(self) -> "SpectroscopyData":
        """Convert to reflectance (R = sample / background or inverse KM/log(1/R)).

        Returns
        -------
        SpectroscopyData
            New instance with ``data_type='reflectance'``.  Returns ``self``
            if already reflectance.

        Raises
        ------
        ValueError
            If ``data_type`` is ``'transmittance'`` or if a background is
            needed but not set.
        """
        if self.data_type == "reflectance":
            return self
        if self.data_type == "single_beam":
            bg = self._require_background()
            if np.any(bg == 0):
                logger.warning("Background contains zeros; reflectance will contain inf")
            return self._convert(self.values / bg, "reflectance")
        if self.data_type == "log_1_r":
            return self._convert(np.power(10.0, -self.values), "reflectance")
        if self.data_type == "absorbance":
            # OMNIC stores DRIFTS/reflectance data as absorbance (= -log₁₀(R)); invert to get R.
            return self._convert(np.power(10.0, -self.values), "reflectance")
        if self.data_type == "kubelka_munk":
            km = self.values
            r = (1.0 + km) - np.sqrt(km * (km + 2.0))
            return self._convert(r, "reflectance")
        raise ValueError(
            f"Cannot convert '{self.data_type}' to reflectance. "
            "Transmittance-based type (transmittance) belongs to a different experiment."
        )

    @records_step
    def to_absorbance(self) -> "SpectroscopyData":
        """Convert to absorbance (A = −log₁₀(T) for transmission experiments).

        Returns
        -------
        SpectroscopyData
            New instance with ``data_type='absorbance'``.  Returns ``self``
            if already absorbance.

        Raises
        ------
        ValueError
            If ``data_type`` is not ``'single_beam'`` or ``'transmittance'``,
            or if a background is needed but not set.
        """
        if self.data_type == "absorbance":
            return self
        if self.data_type == "single_beam":
            bg = self._require_background()
            ratio = self.values / bg
            if np.any(ratio <= 0):
                logger.warning("Non-positive sample/background ratio; absorbance will contain nan/inf")
            return self._convert(-np.log10(ratio), "absorbance")
        if self.data_type == "transmittance":
            if np.any(self.values <= 0):
                logger.warning("Non-positive transmittance; absorbance will contain nan/inf")
            return self._convert(-np.log10(self.values), "absorbance")
        raise ValueError(
            f"Cannot convert '{self.data_type}' to absorbance. "
            "Absorbance is defined as -log₁₀(T) for transmission experiments. "
            "For reflectance experiments use to_log_1_r()."
        )

    @records_step
    def to_log_1_r(self) -> "SpectroscopyData":
        """Convert to log(1/R) for diffuse-reflectance experiments.

        Returns
        -------
        SpectroscopyData
            New instance with ``data_type='log_1_r'``.  Returns ``self``
            if already log(1/R).

        Raises
        ------
        ValueError
            If the current ``data_type`` cannot be converted to log(1/R), or
            if a background is needed but not set.
        """
        if self.data_type == "log_1_r":
            return self
        if self.data_type == "single_beam":
            bg = self._require_background()
            ratio = self.values / bg
            if np.any(ratio <= 0):
                logger.warning("Non-positive sample/background ratio; log(1/R) will contain nan/inf")
            return self._convert(-np.log10(ratio), "log_1_r")
        if self.data_type == "reflectance":
            if np.any(self.values <= 0):
                logger.warning("Non-positive reflectance; log(1/R) will contain nan/inf")
            return self._convert(-np.log10(self.values), "log_1_r")
        if self.data_type == "absorbance":
            # OMNIC stores DRIFTS/reflectance data as absorbance (= -log₁₀(R)); relabel as log(1/R).
            return self._convert(self.values.copy(), "log_1_r")
        if self.data_type == "kubelka_munk":
            km = self.values
            r = (1.0 + km) - np.sqrt(km * (km + 2.0))
            if np.any(r <= 0):
                logger.warning("Non-positive reflectance after KM inversion; log(1/R) will contain nan/inf")
            return self._convert(-np.log10(r), "log_1_r")
        raise ValueError(
            f"Cannot convert '{self.data_type}' to log(1/R). "
            "log(1/R) is defined for reflectance experiments. "
            "For transmission experiments use to_absorbance()."
        )

    @records_step
    def to_kubelka_munk(self) -> "SpectroscopyData":
        """Convert to Kubelka-Munk remission function F(R) = (1−R)² / (2R).

        Returns
        -------
        SpectroscopyData
            New instance with ``data_type='kubelka_munk'``.  Returns ``self``
            if already Kubelka-Munk.

        Raises
        ------
        ValueError
            If the current ``data_type`` is transmittance-based and cannot be
            converted to reflectance, or if a background is needed but not
            set.
        """
        if self.data_type == "kubelka_munk":
            return self
        r_ir = self.to_reflectance()  # raises if transmittance-based
        r = r_ir.values
        if np.any(r <= 0):
            logger.warning("Non-positive reflectance; Kubelka-Munk will contain nan/inf")
        return r_ir._convert((1.0 - r) ** 2 / (2.0 * r), "kubelka_munk")

    def _del_background(self) -> "SpectroscopyData":
        if "background" not in self.ds:
            return self
        return self._new(ds=self.ds.drop_vars("background"))

    def _make_bg_da(self, values: npt.NDArray) -> xr.DataArray:
        return xr.DataArray(
            data=values,
            coords={"wavenumber": self.ds.coords["wavenumber"]},
            dims=["wavenumber"],
            name="background",
            attrs={"data_type": "single_beam"},
        )

    def _set_background(self, bg_single_beam: npt.NDArray) -> "SpectroscopyData":
        return self._new(ds=self.ds.assign({"background": self._make_bg_da(bg_single_beam)}))

    def _switch_background(self, new_bg: npt.NDArray) -> "SpectroscopyData":
        self._warn_drop_baseline("switching background recalculates data values")
        new_values = self._recalculate_with_new_background(new_bg)
        new_attrs = dict(self.ds.attrs)
        ds_new = self._build_ds(self.wavenumber, new_values, tos=self.tos, attrs=new_attrs)
        return self._new(ds=ds_new.assign({"background": self._make_bg_da(new_bg)}))

    def _bg_to_single_beam(
        self, bg_values: npt.NDArray, bg_data_type: Optional[DataType]
    ) -> npt.NDArray:
        """Convert incoming background values to single_beam units.

        If bg_data_type is already single_beam (or None), return as-is.
        Otherwise reconstruct using the existing background (which must exist).
        """
        if bg_data_type is None or bg_data_type == "single_beam":
            return bg_values
        old_bg = self.background
        if old_bg is None:
            raise ValueError(
                f"Cannot convert background from '{bg_data_type}' to single_beam without an "
                "existing background. Assign a single_beam background first, or pass "
                "data_type='single_beam'."
            )
        if bg_data_type in ("transmittance", "reflectance"):
            return bg_values * old_bg
        if bg_data_type in ("absorbance", "log_1_r"):
            return np.power(10.0, -bg_values) * old_bg
        if bg_data_type == "kubelka_munk":
            km = bg_values
            r = (1.0 + km) - np.sqrt(km * (km + 2.0))
            return r * old_bg
        raise ValueError(f"Unknown background data_type '{bg_data_type}'")

    def _recalculate_with_new_background(self, new_bg: npt.NDArray) -> npt.NDArray:
        """Recalculate data values after swapping background (old background must exist)."""
        old_bg = self.background
        dt = self.data_type
        if dt is None:
            raise ValueError("data_type is not set; cannot recalculate values with new background")
        if dt == "single_beam":
            return self.values.copy()
        if dt in ("transmittance", "reflectance"):
            return self.values * (old_bg / new_bg)
        if dt in ("absorbance", "log_1_r"):
            return self.values + np.log10(new_bg / old_bg)
        if dt == "kubelka_munk":
            km = self.values
            r_old = (1.0 + km) - np.sqrt(km * (km + 2.0))
            r_new = r_old * (old_bg / new_bg)
            return (1.0 - r_new) ** 2 / (2.0 * r_new)
        raise ValueError(f"Cannot recalculate values for data_type '{dt}'")

    def _to_data_type(self, data_type: DataType) -> "SpectroscopyData":
        """Convert to a data_type named at runtime (dispatch over the public to_* methods)."""
        converters = {
            "single_beam": "to_single_beam",
            "transmittance": "to_transmittance",
            "reflectance": "to_reflectance",
            "absorbance": "to_absorbance",
            "log_1_r": "to_log_1_r",
            "kubelka_munk": "to_kubelka_munk",
        }
        if data_type not in converters:
            raise ValueError(f"Unknown data_type '{data_type}'; expected one of {list(converters)}")
        return getattr(self, converters[data_type])()
