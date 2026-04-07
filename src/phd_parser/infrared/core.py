from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
import numpy as np
import numpy.typing as npt
from typing import Any, Dict, Literal, Optional

from phd_parser.infrared import omnic
from phd_parser.units import transform_matching_dimensions, transform_wavenumber_frequency, transform_wavenumber_energy

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


X_LABELS = Literal["wavenumber", "frequency", "energy"]
V_LABELS = Literal["absorbance", "transmittance", "reflectance"]



class IRData(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed = True,
        validate_assignment = True
    )

    # X data
    wavenumber: Optional[npt.NDArray] = Field(default=None, description="Wavenumber data in cm^-1")
    wavenumber_2SI_factor: float = Field(default=100.0, description="Factor to convert wavenumber data to SI units (1/cm to 1/m)")
    energy_2SI_factor: float = Field(default=1.0, description="Factor to convert energy data to SI units (default is 1.0, hence energy is expected to be in joules)")
    frequency_2SI_factor: float = Field(default=1.0, description="Factor to convert frequency data to SI units (default is 1.0, hence frequency is expected to be in hertz)")

    # Y data
    tos: Optional[npt.NDArray] = Field(default=None, description="Time of scan data (if available)")
    tos_2SI_factor: float = Field(default=1.0, description="Factor to convert time of scan data to SI units (default is 1.0, hence time of scan is expected to be in seconds)")

    # Values
    values: Optional[npt.NDArray] = Field(default=None, description="Absorbance or transmittance data")
    values_label: V_LABELS = Field(default="absorbance", description="Label for the y data (e.g., 'absorbance', 'transmittance', 'reflectance')")
    values_2SI_factor: float = Field(default=1.0, description="Factor to convert y data to SI units (default is 1.0, hence y is expected to be dimensionless or in a.u.)")
    
    # Metadata
    raw_meta: Dict[str, Any] = Field(default_factory=dict, description="Raw metadata extracted from the file")

    # ================================================
    # Properties
    # ================================================
    @property
    def ndim(self) -> int:
        return self.values.ndim if self.values is not None else 0
    
    @property
    def shape(self) -> Optional[tuple]:
        return self.values.shape if self.values is not None else None
    
    @property
    def size(self) -> Optional[int]:
        return self.values.size if self.values is not None else None

    @property
    def frequency(self) -> Optional[npt.NDArray]:
        if self.wavenumber is not None:
            return transform_wavenumber_frequency(self.wavenumber, wavenumber_2SI_factor=self.wavenumber_2SI_factor, frequency_2SI_factor=self.frequency_2SI_factor, forward=True)
        return None
    
    @property
    def energy(self) -> Optional[npt.NDArray]:
        if self.wavenumber is not None:
            return transform_wavenumber_energy(self.wavenumber, wavenumber_2SI_factor=self.wavenumber_2SI_factor, energy_2SI_factor=self.energy_2SI_factor, forward=True)
        return None
    

    # ================================================
    # helper functions for organizing the data
    # ================================================

    def sort(self, ascending: bool = True) -> None:
        if self.wavenumber is not None and self.values is not None:
            sort_indices = np.argsort(self.wavenumber)
            if not ascending:
                sort_indices = sort_indices[::-1]
            self.wavenumber = self.wavenumber[sort_indices]
            self.values = self.values[:, sort_indices]

    # ================================================
    # reading from omnic .spa files
    # ================================================
    
    @classmethod
    def from_omnic_spa(cls, filepath: str | Path, x_label: X_LABELS = "wavenumber", x_2SI_factor: float = 1.0, v_label: V_LABELS = "absorbance", v_2SI_factor: float = 1.0, tos_2SI_factor: float = 1) -> "IRData":
        # Read the data using omnic module
        raw = omnic.read_spa(filepath)

        if x_label == "wavenumber":
            x = transform_matching_dimensions(raw["data"]["x"], from_2SI_factor=x_2SI_factor, to_2SI_factor=100.0)  # Default is 1/cm, so convert to 1/m if needed
        else:
            logger.error(f"Unsupported x_label '{x_label}'. Only 'wavenumber' is currently supported. Defaulting to 'wavenumber'.")   
            raise NotImplementedError(f"Only 'wavenumber' is currently supported as x_label. Got '{x_label}' instead.")
        
        if v_label in ["absorbance", "transmittance", "reflectance"]:
            v = transform_matching_dimensions(raw["data"]["v"], from_2SI_factor=v_2SI_factor, to_2SI_factor=1.0)  # Default is dimensionless, so convert if needed
        else:
            logger.error(f"Unsupported v_label '{v_label}'. Only 'absorbance', 'transmittance', and 'reflectance' are currently supported. Defaulting to 'absorbance'.")   
            raise NotImplementedError(f"Only 'absorbance', 'transmittance', and 'reflectance' are currently supported as v_label. Got '{v_label}' instead.")
        
        try:
            tos = transform_matching_dimensions(raw["data"]["tos"], from_2SI_factor=tos_2SI_factor, to_2SI_factor=1.0) if "tos" in raw["data"] else None
        except Exception as e:
            logger.warning(f"Failed to transform time of scan data to SI units: {e}. Keeping original values.")
            tos = None

        ir_data = cls(
            wavenumber=np.asarray(x),
            wavenumber_2SI_factor=x_2SI_factor,
            values=np.asarray(v),
            values_label=v_label,
            values_2SI_factor=v_2SI_factor,
            tos=tos,
            tos_2SI_factor=tos_2SI_factor,
            raw_meta=raw["meta"]
        )

        return ir_data
    


    
