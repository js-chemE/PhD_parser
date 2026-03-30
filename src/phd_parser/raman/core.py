from pathlib import Path

from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import numpy.typing as npt
import ramanspy as rp
from phd_parser.raman import btc655n

from typing import Any, Dict, Optional, Literal

class RamanData(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    wavelength_nm: Optional[npt.NDArray] = Field(default=None, description="Wavelength in nanometers")
    wavelength_nm_excitation: Optional[float | int | np.floating | np.integer] = Field(default=None, description="Excitation wavelength in nanometers")
    values: Optional[npt.NDArray] = Field(default=None, description="Raman intensity values")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")

    # ================================================
    # Properties
    # ================================================
    @property
    def ndim(self) -> int:
        return self.values.ndim
    
    @property
    def pixels(self) -> npt.NDArray:
        return np.arange(self.values.size)
    
    @property
    def wavenumber_nm(self) -> npt.NDArray:
        return 1 / self.wavelength_nm
    
    @property
    def wavenumber_cm(self) -> npt.NDArray:
        return 1e7 / self.wavelength_nm
    
    @property
    def shift_nm(self) -> npt.NDArray:
        if self.wavelength_nm_excitation is None:
            raise ValueError("Excitation wavelength is not set, cannot calculate shift.")
        return self.wavelength_nm - self.wavelength_nm_excitation

    @property
    def shift_cm(self) -> npt.NDArray:
        if self.wavelength_nm_excitation is None:
            raise ValueError("Excitation wavelength is not set, cannot calculate shift.")
        return 1e7 / self.wavelength_nm_excitation - self.wavenumber_cm

    # ================================================
    # reading from btc655n export
    # ================================================
    
    @classmethod
    def from_btc655n_export(cls, filepath: str | Path, y_key: btc655n.Y_KEYS = "Raw data #1") -> "RamanData":
        # Read the data using btc655n module
        raw_data = btc655n.read_export(filepath)

        raman = cls(
            wavelength_nm=np.asarray(raw_data["data"]["Wavelength"]),
            wavelength_nm_excitation=raw_data["meta"].get("laser_wavelength"),
            values=np.asarray(raw_data["data"][y_key]),
            meta=raw_data["meta"])
        
        return raman
    

    