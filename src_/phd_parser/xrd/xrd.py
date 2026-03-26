from pydantic import BaseModel, Field, ConfigDict, model_validator

import numpy as np
import numpy.typing as npt


class XRD(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    angle: npt.NDArray[np.float64] = Field(default = np.array([]), description="2-theta angle in degree")
    intensity: npt.NDArray[np.float64] = Field(default = np.array([]), description="Intensity in arbitrary units")
    # ----------------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------------
    @model_validator(mode="after")
    def check_angle_intensity_length(self) -> "XRD":
        if self.angle.size != self.intensity.size:
            raise ValueError(
                f"`angle` and `intensity` must have the same number of entries "
                f"(got angle={self.angle.size}, intensity={self.intensity.size})"
            )
        return self

    # ----------------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Functions
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Reading in
    # ----------------------------------------------------------------------
    @classmethod
    def from_e1290(cls, path: str) -> 'XRD':
        from phd_parser.xrd.xrd_e1290 import read_xy_e1290
        xrd_data = read_xy_e1290(path)
        xrd = cls(
            angle=xrd_data[0],
            intensity=xrd_data[1]
        )
        return xrd