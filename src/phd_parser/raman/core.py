#from pathlib import Path

from pydantic import BaseModel
#import pandas as pd
# import ramanspy as rp
#from phd_parser.raman import btc655n

#from typing import Any, Dict

class RamanData(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True


    # @classmethod
    # def from_btc655n_export(cls, file_path: str | Path) -> "RamanData":
    #     # Read the data using btc655n module
    #     raw_data = btc655n.read_export(file_path)

    #     # Extract the data and metadata
    #     data = raw_data.data
    #     meta = raw_data.meta

    #     return cls(data=data, meta=meta)