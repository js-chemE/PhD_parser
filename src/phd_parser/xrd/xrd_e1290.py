import numpy as np
import numpy.typing as npt
from pathlib import Path

def read_xy_e1290(filename: str | Path, normalize: bool = True) -> npt.NDArray[np.float64]:
    filename = Path(filename)

    xrd = np.loadtxt(filename, skiprows=1, usecols=(0, 1), unpack=True)
    
    if normalize:
        xrd[1] = xrd[1] / xrd[1].max()
    
    return xrd.astype(np.float64)