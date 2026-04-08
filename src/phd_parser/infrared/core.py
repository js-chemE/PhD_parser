from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
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
    values: Optional[npt.NDArray] = Field(default=None, description="Absorbance or transmittance data. If 2D, shape should be (num_scans, num_points)")
    values_label: V_LABELS = Field(default="absorbance", description="Label for the y data (e.g., 'absorbance', 'transmittance', 'reflectance')")
    values_2SI_factor: float = Field(default=1.0, description="Factor to convert y data to SI units (default is 1.0, hence y is expected to be dimensionless or in a.u.)")
    
    # Metadata
    raw_meta: Dict[str, Any] = Field(default_factory=dict, description="Raw metadata extracted from the file")


    # ================================================
    # Validation
    # ================================================
    @model_validator(mode="after")
    def validate_shapes(self) -> "IRData":
        if self.values is None:
            return self

        # ---------------------------
        # Case 1: 1D values
        # ---------------------------
        if self.values.ndim == 1:
            n_points = self.values.size

            if self.wavenumber is not None:
                if self.wavenumber.ndim != 1:
                    raise ValueError("'wavenumber' must be 1D")
                if self.wavenumber.size != n_points:
                    raise ValueError(
                        f"Mismatch: wavenumber size ({self.wavenumber.size}) "
                        f"!= values size ({n_points})"
                    )

            # tos is ignored for 1D
            return self

        # ---------------------------
        # Case 2: 2D values
        # ---------------------------
        elif self.values.ndim == 2:
            n_scans, n_points = self.values.shape

            # wavenumber check
            if self.wavenumber is not None:
                if self.wavenumber.ndim != 1:
                    raise ValueError("'wavenumber' must be 1D")
                if self.wavenumber.size != n_points:
                    raise ValueError(
                        f"Mismatch: wavenumber size ({self.wavenumber.size}) "
                        f"!= values.shape[1] ({n_points})"
                    )

            # tos check
            if self.tos is not None:
                if self.tos.ndim != 1:
                    raise ValueError("'tos' must be 1D")
                if self.tos.size != n_scans:
                    raise ValueError(
                        f"Mismatch: tos size ({self.tos.size}) "
                        f"!= values.shape[0] ({n_scans})"
                    )

            return self

        # ---------------------------
        # Invalid case
        # ---------------------------
        else:
            raise ValueError(
                f"'values' must be 1D or 2D, got shape {self.values.shape}"
            )
    
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
        if self.wavenumber is None or self.values is None:
            return

        sort_indices = np.argsort(self.wavenumber)

        if not ascending:
            sort_indices = sort_indices[::-1]

        self.wavenumber = self.wavenumber[sort_indices]

        if self.values.ndim == 1:
            self.values = self.values[sort_indices]
        elif self.values.ndim == 2:
            self.values = self.values[:, sort_indices]
        else:
            raise ValueError("values must be 1D or 2D")
    # ================================================
    # exporting data
    # ================================================

    def to_csv(self, filepath: str | Path) -> None:
        if self.wavenumber is not None and self.values is not None:
            data_to_save = np.column_stack((self.wavenumber, self.values.T))
            np.savetxt(filepath, data_to_save, delimiter=",", header="wavenumber," + ",".join([f"value_{i}" for i in range(self.values.shape[0])]), comments="")
        else:
            logger.error("Cannot export to CSV: wavenumber or values data is missing.")
            raise ValueError("Cannot export to CSV: wavenumber or values data is missing.")

    # ================================================
    # reading from omnic .spa files
    # ================================================
    
    @classmethod
    def from_omnic_spa(
        cls,
        filepath: str | Path,
        x_label: X_LABELS = "wavenumber",
        x_2SI_factor: float = 1.0,
        v_label: V_LABELS = "absorbance",
        v_2SI_factor: float = 1.0,
        delta_time_seconds: Optional[float] = None) -> "IRData":

        # Read the data using omnic module
        raw = omnic.read_spa(filepath, delta_time_seconds=delta_time_seconds)

        print(f"Raw data keys: {raw.keys()}")
        print(f"Raw data 'data' keys: {raw['data'].keys()}")
        print(f"Raw data 'data' x shape: {raw['data']['x'].shape}")
        print(f"Raw data 'data' tos shape: {raw['data']['tos'].shape if raw['data']['tos'] is not None else None}")
        print(f"Raw data 'data' v shape: {raw['data']['v'].shape}")
        print(f"Raw data meta keys: {raw['meta'].keys()}")


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

        tos = raw["data"]["tos"]

        ir_data = cls(
            wavenumber=np.asarray(x),
            wavenumber_2SI_factor=x_2SI_factor,
            values=np.asarray(v),
            values_label=v_label,
            values_2SI_factor=v_2SI_factor,
            tos=tos,
            tos_2SI_factor=1.0,
            raw_meta=raw["meta"]
        )

        return ir_data