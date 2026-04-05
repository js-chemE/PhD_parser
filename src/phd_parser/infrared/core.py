from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
import pandas as pd
import numpy as np
import numpy.typing as npt
import ramanspy as rp
from phd_parser.infrared import omnic

from typing import Any, Dict, Optional, Literal

class IRData(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed = True,
        validate_assignment = True
    )

    x: Optional[npt.NDArray] = Field(default=None, description="X-axis data (e.g., wavenumber)")
    y: Optional[npt.NDArray] = Field(default=None, description="Y-axis data (e.g., absorbance)")
    raw_meta: Dict[str, Any] = Field(default_factory=dict, description="Raw metadata extracted from the file")

    # ================================================
    # Properties
    # ================================================
    @property
    def ndim(self) -> int:
        return self.y.ndim if self.y is not None else 0
    

    # ================================================
    # reading from omnic .spa files
    # ================================================
    
    @classmethod
    def from_omnic_spa(cls, filepath: str | Path) -> "IRData":
        # Read the data using omnic module
        raw = omnic.read_spa(filepath)
    
        ir_data = cls(
            x=np.asarray(raw["data"]["x"]),
            y=np.asarray(raw["data"]["y"]),
            raw_meta=raw["meta"])
        
        return ir_data