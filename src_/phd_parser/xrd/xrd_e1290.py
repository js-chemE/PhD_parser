import numpy as np

def read_xy_e1290(filename: str, normalize: bool = True) -> np.ndarray:
    xrd = np.loadtxt(filename, skiprows=1, usecols=(0, 1), unpack=True)
    if normalize:
        xrd[1] = xrd[1] / xrd[1].max()
    return xrd