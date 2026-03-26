from pydantic import BaseModel, ConfigDict, Field
import numpy as np
import numpy.typing as npt

from typing import Union

class TGA(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    temperature: npt.NDArray[np.float64]
    mass: npt.NDArray[np.float64]
    mass_init: float | int | None = Field(
        default=None,
    )
    baseline: Union["TGA", None] = Field(
        default=None,
    )
    # Back Up
    backup_temperature: npt.NDArray[np.float64] | None = Field(
        default=None,
    )
    backup_mass: npt.NDArray[np.float64] | None = Field(
        default=None,
    )

    # ----------------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------------

    @property
    def mass_fraction(self) -> npt.NDArray[np.float64]:
        if self.mass_init is None:
            return np.zeros_like(self.mass)
        else:
            return self.mass / self.mass_init

    @property
    def derivative(self) -> npt.NDArray[np.float64]:
        return np.gradient(self.mass, self.temperature)
    
    @property
    def derivative_fraction(self) -> npt.NDArray[np.float64]:
        return np.gradient(self.mass_fraction, self.temperature)
    
    # ----------------------------------------------------------------------
    # Functions
    # ----------------------------------------------------------------------

    def backup(self) -> None:
        self.backup_temperature = self.temperature.copy()
        self.backup_mass = self.mass.copy()

    def restore(self) -> None:
        if self.backup_temperature is not None:
            self.temperature = self.backup_temperature
        if self.backup_mass is not None:
            self.mass = self.backup_mass

    def cut_front(self, index: int | None = None, temperature: float | None = None) -> None:
        if index is None and temperature is None:
            raise ValueError("Either index or temperature must be provided.")
        elif index is not None:
            self.temperature = self.temperature[index:]
            self.mass = self.mass[index:]
        elif temperature is not None:
            self.temperature = self.temperature[self.temperature >= temperature]
            self.mass = self.mass[self.temperature >= temperature]
        else:
            raise ValueError("Either index or temperature must be provided.")

    def cut_back(self, index: int | None = None, temperature: float | None = None) -> None:
        if index is None and temperature is None:
            raise ValueError("Either index or temperature must be provided.")
        elif index is not None:
            self.temperature = self.temperature[:index]
            self.mass = self.mass[:index]
        elif temperature is not None:
            self.temperature = self.temperature[self.temperature <= temperature]
            self.mass = self.mass[self.temperature <= temperature]
        else:
            raise ValueError("Either index or temperature must be provided.")

    def correct(self, baseline: "TGA", backup: bool = False) -> None:
        if baseline is None:
            raise ValueError("Baseline TGA must be provided.")
        if backup:
            self.backup()
        self.mass -= np.interp(self.temperature, baseline.temperature, baseline.mass)
    
    def smooth(self, window_length: int = 11, polyorder: int = 2) -> None:
        """Return a smoothed TGA object using Savitzky–Golay filter."""
        from scipy.signal import savgol_filter
        self.mass = savgol_filter(self.mass, window_length, polyorder)


    # ----------------------------------------------------------------------
    # Reading in
    # ----------------------------------------------------------------------

    @classmethod
    def from_e2290(cls, path: str, baseline_path: str | None = None, in_kelvin: bool = True) -> "TGA":
        from phd_parser.tga.tga_e2290 import read_tga_e2290

        correction = 0.0
        if in_kelvin:
            correction = 273.15
        
        e2290 = read_tga_e2290(path)
        tga = cls(
            temperature=e2290["data"]["Ts"].values + correction,
            mass=e2290["data"]["Value"].values,
            mass_init=e2290["weight"]
        )
        tga.backup()

        if baseline_path is not None:
            e2290_baseline = read_tga_e2290(baseline_path)
            baseline = cls(
                temperature=e2290_baseline["data"]["Ts"].values + correction,
                mass=e2290_baseline["data"]["Value"].values,
                mass_init=e2290_baseline["weight"]
            )
            tga.correct(baseline, backup=False)
        return tga