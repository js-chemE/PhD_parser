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

Branch = Literal["ads", "des"]


class PhysisorptionData(BaseModel):
    """Minimal, instrument-agnostic gas-physisorption isotherm container.

    Only the raw isotherm is modelled here — relative pressure versus
    quantity adsorbed, per branch — since that is the one thing every
    physisorption instrument and analysis method (BET, t-Plot, BJH, ...)
    agrees on. All processing methods are immutable and return a new
    ``PhysisorptionData`` instance.

    BET surface-area results are also modelled directly, since virtually
    every physisorption analysis derives them from the isotherm regardless
    of instrument. Other instrument-specific analyses (t-Plot, BJH, ...)
    are not modelled yet and stay under :attr:`report`.

    Attributes
    ----------
    ds : xr.Dataset
        Dataset with up to two 1-D data variables, ``"q_ads"`` (dims
        ``("p_rel_ads",)``) and ``"q_des"`` (dims ``("p_rel_des",)``), in
        cm³/g STP. The ``p_rel_*`` dims double as their own float
        coordinate (relative pressure P/P₀, ascending). ``ds.attrs`` may
        hold a ``"bet"`` dict (see :attr:`bet`) and instrument-specific
        provenance under the ``"report"`` key; nothing else is interpreted
        by this class.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    ds: xr.Dataset = Field(
        description=(
            "Dataset with 'q_ads' (dims ('p_rel_ads',)) and/or 'q_des' "
            "(dims ('p_rel_des',)); p_rel_* dims are float coordinates."
        )
    )

    # ----------------------------------------------------------------
    # Validators
    # ----------------------------------------------------------------

    @field_validator("ds", mode="before")
    @classmethod
    def _validate_ds(cls, v: Any) -> xr.Dataset:
        if not isinstance(v, xr.Dataset):
            raise TypeError(f"'ds' must be an xr.Dataset, got {type(v)}")
        present = [b for b in ("ads", "des") if f"q_{b}" in v]
        if not present:
            raise ValueError("Dataset must contain 'q_ads' and/or 'q_des'.")
        for branch in present:
            da = v[f"q_{branch}"]
            dim = f"p_rel_{branch}"
            if da.dims != (dim,):
                raise ValueError(f"'q_{branch}' must have dims ('{dim}',), got {da.dims}")
        return v

    # ----------------------------------------------------------------
    # Core properties
    # ----------------------------------------------------------------

    @property
    def branches(self) -> list[Branch]:
        """Adsorption/desorption branches present in the dataset.

        Returns
        -------
        list of {"ads", "des"}
            Branches for which isotherm data is available.
        """
        return [b for b in ("ads", "des") if f"q_{b}" in self.ds]

    def _check_branch(self, branch: Branch) -> None:
        if branch not in self.branches:
            raise KeyError(f"Branch '{branch}' not found. Available: {self.branches}")

    def p_rel(self, branch: Branch = "ads") -> npt.NDArray:
        """Relative-pressure (P/P₀) coordinate for one branch.

        Parameters
        ----------
        branch : {"ads", "des"}, optional
            Isotherm branch to query (default is ``"ads"``).

        Returns
        -------
        numpy.ndarray
            1-D float array, ascending.

        Raises
        ------
        KeyError
            If ``branch`` is not present in the dataset.
        """
        self._check_branch(branch)
        return self.ds.coords[f"p_rel_{branch}"].values

    def q(self, branch: Branch = "ads") -> npt.NDArray:
        """Quantity adsorbed (cm³/g STP) for one branch.

        Parameters
        ----------
        branch : {"ads", "des"}, optional
            Isotherm branch to query (default is ``"ads"``).

        Returns
        -------
        numpy.ndarray
            1-D float array aligned with :meth:`p_rel`.

        Raises
        ------
        KeyError
            If ``branch`` is not present in the dataset.
        """
        self._check_branch(branch)
        return self.ds[f"q_{branch}"].values

    def n_points(self, branch: Branch = "ads") -> int:
        """Number of isotherm points on one branch.

        Parameters
        ----------
        branch : {"ads", "des"}, optional
            Isotherm branch to query (default is ``"ads"``).

        Returns
        -------
        int
            Number of points on the branch.

        Raises
        ------
        KeyError
            If ``branch`` is not present in the dataset.
        """
        return self.p_rel(branch).size

    @property
    def report(self) -> dict[str, Any]:
        """Raw, instrument-specific report data preserved for provenance.

        Returns
        -------
        dict
            Whatever the loading parser stashed under ``ds.attrs["report"]``
            (e.g. t-Plot/BJH results and the sample log for a TriStar
            export), or an empty dict if none was provided. Stored as a
            JSON string on the dataset so it survives :meth:`to_netcdf`;
            decoded transparently here.
        """
        return _decode_json_attr(self.ds.attrs.get("report")) or {}

    @property
    def bet(self) -> Optional[dict[str, Any]]:
        """BET fit results, if available.

        Returns
        -------
        dict or None
            Dict with keys ``"surface_area"`` (m²/g), ``"surface_area_error"``,
            ``"slope"``, ``"slope_error"``, ``"y_intercept"``,
            ``"y_intercept_error"``, ``"c_constant"``, ``"qm"``,
            ``"correlation_coefficient"``, ``"cross_sectional_area"``
            (with ``*_unit`` siblings where applicable), or ``None`` if no
            BET results were supplied at construction time.
        """
        return _decode_json_attr(self.ds.attrs.get("bet"))

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

    def get_quantity(
        self,
        target_p_rel: Union[float, Sequence[float]],
        branch: Branch = "ads",
        method: Literal["nearest", "linear"] = "nearest",
        tolerance: Optional[float] = 0.02,
    ) -> Union[float, npt.NDArray]:
        """Quantity adsorbed at one or more relative-pressure values.

        Parameters
        ----------
        target_p_rel : float or sequence of float
            Target P/P₀ value(s).
        branch : {"ads", "des"}, optional
            Isotherm branch to query (default is ``"ads"``).
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
        KeyError
            If ``branch`` is not present in the dataset.
        ValueError
            If any requested value is farther than ``tolerance`` from the
            nearest grid point.
        """
        self._check_branch(branch)
        grid = self.p_rel(branch)
        scalar_input = np.ndim(target_p_rel) == 0
        targets = [float(target_p_rel)] if scalar_input else [float(t) for t in target_p_rel]

        if tolerance is not None:
            for t in targets:
                dist = float(np.abs(grid - t).min())
                if dist > tolerance:
                    raise ValueError(
                        f"Requested p_rel {t:.4f} is {dist:.4f} from the nearest point "
                        f"(tolerance: {tolerance:.4f})"
                    )

        result = self.ds[f"q_{branch}"].sel(
            **{f"p_rel_{branch}": targets if not scalar_input else targets[0]}, method=method
        )
        return float(result) if scalar_input else result.values

    # ----------------------------------------------------------------
    # Export
    # ----------------------------------------------------------------

    def to_netcdf(self, filepath: Union[str, Path]) -> None:
        """Save the dataset to a NetCDF4 file.

        The saved file can be reloaded with :meth:`from_netcdf`.

        Parameters
        ----------
        filepath : str or Path
            Destination file path.
        """
        self.ds.to_netcdf(filepath)
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
        return cls(ds=xr.open_dataset(filepath))

    # ----------------------------------------------------------------
    # Private: dataset builder
    # ----------------------------------------------------------------

    @staticmethod
    def _build_ds(
        p_rel_ads: Optional[npt.NDArray] = None,
        q_ads: Optional[npt.NDArray] = None,
        p_rel_des: Optional[npt.NDArray] = None,
        q_des: Optional[npt.NDArray] = None,
        bet: Optional[dict[str, Any]] = None,
        attrs: Optional[dict[str, Any]] = None,
    ) -> xr.Dataset:
        data_vars: dict[str, xr.DataArray] = {}
        if p_rel_ads is not None and q_ads is not None:
            data_vars["q_ads"] = xr.DataArray(
                np.asarray(q_ads, dtype=float),
                coords={"p_rel_ads": np.asarray(p_rel_ads, dtype=float)},
                dims=["p_rel_ads"],
            )
        if p_rel_des is not None and q_des is not None:
            data_vars["q_des"] = xr.DataArray(
                np.asarray(q_des, dtype=float),
                coords={"p_rel_des": np.asarray(p_rel_des, dtype=float)},
                dims=["p_rel_des"],
            )
        ds_attrs = dict(attrs or {})
        if bet is not None:
            ds_attrs["bet"] = bet
        for key in ("report", "bet"):
            if isinstance(ds_attrs.get(key), dict):
                ds_attrs[key] = json.dumps(ds_attrs[key], default=str)
        return xr.Dataset(data_vars, attrs=ds_attrs)

    # ----------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        p_rel_ads: Optional[npt.NDArray] = None,
        q_ads: Optional[npt.NDArray] = None,
        p_rel_des: Optional[npt.NDArray] = None,
        q_des: Optional[npt.NDArray] = None,
        bet: Optional[dict[str, Any]] = None,
        attrs: Optional[dict[str, Any]] = None,
    ) -> "PhysisorptionData":
        """Construct a ``PhysisorptionData`` directly from isotherm arrays.

        At least one branch (``p_rel_ads``/``q_ads`` or ``p_rel_des``/``q_des``)
        must be provided.

        Parameters
        ----------
        p_rel_ads : numpy.ndarray or None, optional
            Adsorption-branch relative pressure (P/P₀) values.
        q_ads : numpy.ndarray or None, optional
            Adsorption-branch quantity adsorbed (cm³/g STP), same length
            as ``p_rel_ads``.
        p_rel_des : numpy.ndarray or None, optional
            Desorption-branch relative pressure (P/P₀) values.
        q_des : numpy.ndarray or None, optional
            Desorption-branch quantity adsorbed (cm³/g STP), same length
            as ``p_rel_des``.
        bet : dict or None, optional
            BET fit results to store under :attr:`bet` (see that property
            for the expected keys).
        attrs : dict or None, optional
            Additional attributes to store on the dataset (e.g. a
            ``"report"`` key with instrument-specific provenance).

        Returns
        -------
        PhysisorptionData
            New instance built from the supplied arrays.
        """
        ds = cls._build_ds(p_rel_ads, q_ads, p_rel_des, q_des, bet, attrs)
        return cls(ds=ds)

    @classmethod
    def from_tristar_xls(cls, filepath: Union[str, Path]) -> "PhysisorptionData":
        """Load a ``PhysisorptionData`` from a TriStar II 3020 multi-report ``.XLS`` export.

        The isotherm and BET fit results are modelled directly; the rest of
        the parsed report (t-Plot, BJH, sample log, ...) is preserved
        verbatim under :attr:`report` for later use.

        Parameters
        ----------
        filepath : str or Path
            Path to the ``.XLS`` file.

        Returns
        -------
        PhysisorptionData
            New instance populated with the parsed isotherm and BET results.

        Raises
        ------
        ValueError
            If the file has no recognisable Isotherm Tabular Report section.
        """
        parsed = tristar.read_export(filepath)
        data = parsed["data"]
        if not data:
            raise ValueError(f"No isotherm data could be extracted from {filepath}")
        return cls.from_arrays(**data, bet=parsed.get("bet"), attrs={"report": parsed["meta"]})

    # ----------------------------------------------------------------
    # Dunder
    # ----------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"{b}={self.n_points(b)} pts" for b in self.branches]
        if self.surface_area_bet is not None:
            parts.append(f"BET={self.surface_area_bet:.2f} m\xb2/g")
        return f"PhysisorptionData({', '.join(parts)})"
