from pydantic import BaseModel, Field
import pandas as pd

from typing import Any, Dict

class RamanData(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    data: pd.DataFrame = Field(..., description="DataFrame containing the Raman data")
    meta: Dict[str, Any] = Field(..., description="Dictionary containing the metadata")