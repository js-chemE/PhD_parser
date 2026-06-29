from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, field_validator

import phd_parser.physisorption.tristar as tristar

logger = logging.getLogger(__name__)


def _decode_json_attr(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


class PhysisorptionData(BaseModel):
    """A single gas-physisorption isotherm branch (adsorption or desorption).

    Wraps one ``xarray.DataArray`` of quantity adsorbed versus relative
    pressure — the one thing every physisorption instrument and analysis
    method (BET, t-Plot, BJH, ...) agrees on — exactly the way a single
    Raman spectrum or XRD pattern is wrapped elsewhere in this repo: the
    physical axis (relative pressure) is the array's own dimension, and
    there is no separate "branch" concept baked into the type. A reading
    that produces both branches (e.g. :meth:`from_tristar_xls`) returns two
    separate instances.

    BET surface-area results are modelled directly, since they are derived
    straight from this curve (conventionally the adsorption branch). Other
    instrument-specific analyses (t-Plot, BJH, sample log, ...) are not
    modelled yet and stay under :attr:`report`.

    Attributes
    ----------
    quantity_adsorbed : xr.DataArray
        Quantity adsorbed in cm³/g STP, with dims ``("relative_pressure",)``.
        The ``relative_pressure`` dimension doubles as its own float
        coordinate (P/P₀, ascending). ``.attrs`` may hold a ``"bet"`` dict
        (see :attr:`bet`) and instrument-specific provenance under the
        ``"report"`` key; nothing else is interpreted by this class.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    quantity_adsorbed: xr.DataArray = Field(
        description="Quantity adsorbed (cm³/g STP), dims=('relative_pressure',)."
    )

    # ----------------------------------------------------------------
    # Validators
    # ----------------------------------------------------------------

    @field_validator("quantity_adsorbed", mode="before")
    @classmethod
    def _validate_quantity_adsorbed(cls, v: Any) -> xr.DataArray:
        if not isinstance(v, xr.DataArray):
            raise TypeError(f"'quantity_adsorbed' must be an xr.DataArray, got {type(v)}")
        if v.dims != ("relative_pressure",):
            raise ValueError(
                f"'quantity_adsorbed' must have dims ('relative_pressure',), got {v.dims}"
            )
        return v

    # ----------------------------------------------------------------
    # Core properties
    # ----------------------------------------------------------------

    @property
    def relative_pressure(self) -> npt.NDArray:
        """Relative pressure (P/P₀) coordinate.

        Returns
        -------
        numpy.ndarray
            1-D float array, ascending.
        """
        return self.quantity_adsorbed.coords["relative_pressure"].values

    @property
    def values(self) -> npt.NDArray:
        """Quantity adsorbed (cm³/g STP).

        Returns
        -------
        numpy.ndarray
            1-D float array aligned with :attr:`relative_pressure`.
        """
        return self.quantity_adsorbed.values

    @property
    def n_points(self) -> int:
        """Number of isotherm points on this branch.

        Returns
        -------
        int
            Number of points.
        """
        return self.relative_pressure.size

    @property
    def report(self) -> dict[str, Any]:
        """Raw, instrument-specific report data preserved for provenance.

        Returns
        -------
        dict
            Whatever the loading parser stashed under
            ``quantity_adsorbed.attrs["report"]`` (e.g. t-Plot/BJH results
            and the sample log for a TriStar export), or an empty dict if
            none was provided. Stored as a JSON string internally so it
            survives :meth:`to_netcdf`; decoded transparently here.
        """
        return _decode_json_attr(self.quantity_adsorbed.attrs.get("report")) or {}

    @property
    def bet(self) -> Optional[dict[str, Any]]:
        """BET fit results, if available.

        Returns
        -------
        dict or None
            Dict with keys ``"surface_area"`` (m²/g), ``"surface_area_error"``,
            ``"slope"``, ``"slope_error"``, ``"y_intercept"``,
            ``"y_intercept_error"``, ``"c_constant"``, ``"monolayer_capacity"``,
            ``"correlation_coefficient"``, ``"cross_sectional_area"``
            (with ``*_unit`` siblings where applicable), or ``None`` if no
            BET results were supplied at construction time.
        """
        return _decode_json_attr(self.quantity_adsorbed.attrs.get("bet"))

    @property
    def surface_area_bet(self) -> Optional[float]:
        """BET specific surface area in m²/g.

        Returns
        -------
        float or None
            ``self.bet["surface_area"]``, or ``None`` if :attr:`bet` is
            unset.
        """
        return self.bet["surface_area"] if self.bet else None

    # ----------------------------------------------------------------
    # Lookups
    # ----------------------------------------------------------------

    def get_quantity_adsorbed(
        self,
        target_relative_pressure: Union[float, Sequence[float]],
        method: Literal["nearest", "linear"] = "nearest",
        tolerance: Optional[float] = 0.02,
    ) -> Union[float, npt.NDArray]:
        """Quantity adsorbed at one or more relative-pressure values.

        Parameters
        ----------
        target_relative_pressure : float or sequence of float
            Target P/P₀ value(s).
        method : {"nearest", "linear"}, optional
            xarray selection method (default is ``"nearest"``).
        tolerance : float or None, optional
            Maximum allowed distance between a requested value and the
            nearest grid point. Raises ``ValueError`` if exceeded. Default
            is ``0.02``.

        Returns
        -------
        float or numpy.ndarray
            A plain ``float`` for scalar input, otherwise a 1-D array.

        Raises
        ------
        ValueError
            If any requested value is farther than ``tolerance`` from the
            nearest grid point.
        """
        grid = self.relative_pressure
        scalar_input = np.ndim(target_relative_pressure) == 0
        targets = (
            [float(target_relative_pressure)]
            if scalar_input
            else [float(t) for t in target_relative_pressure]
        )

        if tolerance is not None:
            for t in targets:
                dist = float(np.abs(grid - t).min())
                if dist > tolerance:
                    raise ValueError(
                        f"Requested relative pressure {t:.4f} is {dist:.4f} from the "
                        f"nearest point (tolerance: {tolerance:.4f})"
                    )

        result = self.quantity_adsorbed.sel(
            relative_pressure=targets if not scalar_input else targets[0], method=method
        )
        return float(result) if scalar_input else result.values

    # ----------------------------------------------------------------
    # Export
    # ----------------------------------------------------------------

    def to_netcdf(self, filepath: Union[str, Path]) -> None:
        """Save the isotherm to a NetCDF4 file.

        The saved file can be reloaded with :meth:`from_netcdf`.

        Parameters
        ----------
        filepath : str or Path
            Destination file path.
        """
        self.quantity_adsorbed.to_netcdf(filepath)
        logger.debug("Saved NetCDF → %s", filepath)

    @classmethod
    def from_netcdf(cls, filepath: Union[str, Path]) -> "PhysisorptionData":
        """Reload a ``PhysisorptionData`` previously saved with :meth:`to_netcdf`.

        Parameters
        ----------
        filepath : str or Path
            Path to the NetCDF4 file.

        Returns
        -------
        PhysisorptionData
            Instance reconstructed from the file.
        """
        return cls(quantity_adsorbed=xr.open_dataarray(filepath))

    # ----------------------------------------------------------------
    # Private: array builder
    # ----------------------------------------------------------------

    @staticmethod
    def _build_quantity_adsorbed(
        relative_pressure: npt.NDArray,
        quantity_adsorbed: npt.NDArray,
        bet: Optional[dict[str, Any]] = None,
        attrs: Optional[dict[str, Any]] = None,
    ) -> xr.DataArray:
        da_attrs = dict(attrs or {})
        if bet is not None:
            da_attrs["bet"] = bet
        for key in ("report", "bet"):
            if isinstance(da_attrs.get(key), dict):
                da_attrs[key] = json.dumps(da_attrs[key], default=str)
        return xr.DataArray(
            np.asarray(quantity_adsorbed, dtype=float),
            coords={"relative_pressure": np.asarray(relative_pressure, dtype=float)},
            dims=["relative_pressure"],
            name="quantity_adsorbed",
            attrs=da_attrs,
        )

    # ----------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        relative_pressure: npt.NDArray,
        quantity_adsorbed: npt.NDArray,
        bet: Optional[dict[str, Any]] = None,
        attrs: Optional[dict[str, Any]] = None,
    ) -> "PhysisorptionData":
        """Construct a ``PhysisorptionData`` directly from isotherm arrays.

        Parameters
        ----------
        relative_pressure : numpy.ndarray
            Relative pressure (P/P₀) values.
        quantity_adsorbed : numpy.ndarray
            Quantity adsorbed (cm³/g STP), same length as
            ``relative_pressure``.
        bet : dict or None, optional
            BET fit results to store under :attr:`bet` (see that property
            for the expected keys).
        attrs : dict or None, optional
            Additional attributes to store on the array (e.g. a
            ``"report"`` key with instrument-specific provenance).

        Returns
        -------
        PhysisorptionData
            New instance built from the supplied arrays.
        """
        da = cls._build_quantity_adsorbed(relative_pressure, quantity_adsorbed, bet, attrs)
        return cls(quantity_adsorbed=da)

    @classmethod
    def from_tristar_xls(cls, filepath: Union[str, Path]) -> dict[str, "PhysisorptionData"]:
        """Load isotherm branches from a TriStar II 3020 multi-report ``.XLS`` export.

        The isotherm and BET fit results are modelled directly on the
        adsorption branch; the rest of the parsed report (t-Plot, BJH,
        sample log, ...) is preserved verbatim under :attr:`report` on
        every returned branch for later use.

        Parameters
        ----------
        filepath : str or Path
            Path to the ``.XLS`` file.

        Returns
        -------
        dict of str to PhysisorptionData
            Dict with up to two keys, ``"adsorption"`` and ``"desorption"``,
            for whichever branches were present in the export.

        Raises
        ------
        ValueError
            If the file has no recognisable Isotherm Tabular Report section.
        """
        parsed = tristar.read_export(filepath)
        data = parsed["data"]
        if not data:
            raise ValueError(f"No isotherm data could be extracted from {filepath}")

        attrs = {"report": parsed["meta"]}
        branches: dict[str, "PhysisorptionData"] = {}
        if "adsorption" in data:
            branches["adsorption"] = cls.from_arrays(
                **data["adsorption"], bet=parsed.get("bet"), attrs=attrs
            )
        if "desorption" in data:
            branches["desorption"] = cls.from_arrays(**data["desorption"], attrs=attrs)
        return branches

    # ----------------------------------------------------------------
    # Dunder
    # ----------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"n_points={self.n_points}"]
        if self.surface_area_bet is not None:
            parts.append(f"BET={self.surface_area_bet:.2f} m\xb2/g")
        return f"PhysisorptionData({', '.join(parts)})"
