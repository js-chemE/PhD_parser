from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
import numpy as np
import numpy.typing as npt
import logging

from typing import Optional, Union

logger = logging.getLogger(__name__)


class TGAData(BaseModel):
    """Pydantic model for thermogravimetric analysis (TGA) data.

    Wraps paired temperature and mass arrays and provides immutable
    processing methods.  All transformations return a new ``TGAData``
    instance rather than mutating ``self``.

    Parameters
    ----------
    temperature : ndarray of float64
        Temperature values in Kelvin.
    mass : ndarray of float64
        Sample mass values in milligrams.
    mass_init : float or int, optional
        Initial sample mass in milligrams used to compute
        :attr:`mass_fraction`.  When ``None`` the fraction is zero.
    baseline : TGAData, optional
        Baseline dataset stored for provenance after :meth:`correct` is
        called.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    temperature: npt.NDArray[np.float64] = Field(
        description="Temperature array (K by default)"
    )
    mass: npt.NDArray[np.float64] = Field(
        description="Mass array (mg)"
    )
    mass_init: float | int | None = Field(
        default=None, description="Initial sample mass for computing mass fraction"
    )
    baseline: Union["TGAData", None] = Field(
        default=None, description="Baseline used for correction, stored for provenance"
    )

    # ----------------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------------

    @property
    def mass_fraction(self) -> npt.NDArray[np.float64]:
        """Mass normalised by the initial sample mass.

        Returns
        -------
        ndarray of float64
            Element-wise ``mass / mass_init``.  Returns an array of
            zeros when :attr:`mass_init` is ``None``.
        """
        if self.mass_init is None:
            return np.zeros_like(self.mass)
        return self.mass / self.mass_init

    @property
    def derivative(self) -> npt.NDArray[np.float64]:
        """Numerical derivative of mass with respect to temperature (dm/dT).

        Returns
        -------
        ndarray of float64
            Gradient of :attr:`mass` along :attr:`temperature` in mg K⁻¹.
        """
        return np.gradient(self.mass, self.temperature)

    @property
    def derivative_fraction(self) -> npt.NDArray[np.float64]:
        """Numerical derivative of mass fraction with respect to temperature.

        Returns
        -------
        ndarray of float64
            Gradient of :attr:`mass_fraction` along :attr:`temperature`
            in K⁻¹.
        """
        return np.gradient(self.mass_fraction, self.temperature)

    # ----------------------------------------------------------------------
    # Immutable processing — all return a new TGAData
    # ----------------------------------------------------------------------

    def cut_front(self, index: int | None = None, temperature: float | None = None) -> "TGAData":
        """Remove data points before a given index or temperature.

        Exactly one of ``index`` or ``temperature`` must be supplied.

        Parameters
        ----------
        index : int, optional
            Integer array index; all points at positions ``>= index``
            are kept.
        temperature : float, optional
            Temperature threshold in Kelvin; all points with
            ``temperature >= threshold`` are kept.

        Returns
        -------
        TGAData
            New instance containing only the retained data points.

        Raises
        ------
        ValueError
            When neither ``index`` nor ``temperature`` is provided.
        """
        if index is not None:
            return TGAData(
                temperature=self.temperature[index:],
                mass=self.mass[index:],
                mass_init=self.mass_init,
                baseline=self.baseline,
            )
        if temperature is not None:
            mask = self.temperature >= temperature
            return TGAData(
                temperature=self.temperature[mask],
                mass=self.mass[mask],
                mass_init=self.mass_init,
                baseline=self.baseline,
            )
        raise ValueError("Either index or temperature must be provided.")

    def cut_back(self, index: int | None = None, temperature: float | None = None) -> "TGAData":
        """Remove data points after a given index or temperature.

        Exactly one of ``index`` or ``temperature`` must be supplied.

        Parameters
        ----------
        index : int, optional
            Integer array index; all points at positions ``< index``
            are kept.
        temperature : float, optional
            Temperature threshold in Kelvin; all points with
            ``temperature <= threshold`` are kept.

        Returns
        -------
        TGAData
            New instance containing only the retained data points.

        Raises
        ------
        ValueError
            When neither ``index`` nor ``temperature`` is provided.
        """
        if index is not None:
            return TGAData(
                temperature=self.temperature[:index],
                mass=self.mass[:index],
                mass_init=self.mass_init,
                baseline=self.baseline,
            )
        if temperature is not None:
            mask = self.temperature <= temperature
            return TGAData(
                temperature=self.temperature[mask],
                mass=self.mass[mask],
                mass_init=self.mass_init,
                baseline=self.baseline,
            )
        raise ValueError("Either index or temperature must be provided.")

    def correct(self, baseline: "TGAData") -> "TGAData":
        """Subtract a baseline measurement interpolated onto the sample temperature axis.

        The baseline mass is linearly interpolated to match
        :attr:`temperature` before subtraction.

        Parameters
        ----------
        baseline : TGAData
            Blank or reference measurement to subtract.

        Returns
        -------
        TGAData
            New instance with corrected mass and ``baseline`` stored
            for provenance.
        """
        corrected_mass = self.mass - np.interp(
            self.temperature, baseline.temperature, baseline.mass
        )
        logger.debug("Baseline correction applied.")
        return TGAData(
            temperature=self.temperature.copy(),
            mass=corrected_mass,
            mass_init=self.mass_init,
            baseline=baseline,
        )

    def smooth(self, window_length: int = 11, polyorder: int = 2) -> "TGAData":
        """Smooth the mass signal with a Savitzky-Golay filter.

        Parameters
        ----------
        window_length : int, optional
            Length of the filter window in number of data points
            (default is 11).
        polyorder : int, optional
            Polynomial order used to fit within each window
            (default is 2).

        Returns
        -------
        TGAData
            New instance with the smoothed mass array.
        """
        from scipy.signal import savgol_filter
        return TGAData(
            temperature=self.temperature.copy(),
            mass=savgol_filter(self.mass, window_length, polyorder),
            mass_init=self.mass_init,
            baseline=self.baseline,
        )

    # ----------------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------------

    @classmethod
    def from_e2290(
        cls,
        path: str | Path,
        baseline_path: Optional[str | Path] = None,
        in_kelvin: bool = True,
    ) -> "TGAData":
        """Load TGA data from a Mettler Toledo E2290 ``.txt`` export.

        Parameters
        ----------
        path : str or Path
            Path to the E2290 ``.txt`` export file.
        baseline_path : str or Path, optional
            Path to a blank/reference E2290 file.  When provided the
            baseline is automatically subtracted via :meth:`correct`.
        in_kelvin : bool, optional
            When ``True`` (default) temperature values are converted from
            °C to K by adding 273.15.

        Returns
        -------
        TGAData
            Parsed (and optionally baseline-corrected) dataset.
        """
        from phd_parser.tga.e2290 import read_export

        path = Path(path)
        correction = 273.15 if in_kelvin else 0.0

        e2290 = read_export(path)
        tga = cls(
            temperature=e2290["data"]["Ts"].values + correction,
            mass=e2290["data"]["Value"].values,
            mass_init=e2290["weight"],
        )

        if baseline_path is not None:
            e2290_baseline = read_export(Path(baseline_path))
            baseline = cls(
                temperature=e2290_baseline["data"]["Ts"].values + correction,
                mass=e2290_baseline["data"]["Value"].values,
                mass_init=e2290_baseline["weight"],
            )
            tga = tga.correct(baseline)

        return tga
