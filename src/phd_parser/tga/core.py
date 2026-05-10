from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
import numpy as np
import numpy.typing as npt
import logging

from typing import Optional, Union

logger = logging.getLogger(__name__)


class TGAData(BaseModel):
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
        if self.mass_init is None:
            return np.zeros_like(self.mass)
        return self.mass / self.mass_init

    @property
    def derivative(self) -> npt.NDArray[np.float64]:
        return np.gradient(self.mass, self.temperature)

    @property
    def derivative_fraction(self) -> npt.NDArray[np.float64]:
        return np.gradient(self.mass_fraction, self.temperature)

    # ----------------------------------------------------------------------
    # Immutable processing — all return a new TGAData
    # ----------------------------------------------------------------------

    def cut_front(self, index: int | None = None, temperature: float | None = None) -> "TGAData":
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
