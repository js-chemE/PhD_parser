import numpy as np
import numpy.typing as npt
from pathlib import Path

def read_xy_e1290(filename: str | Path, normalize: bool = True) -> npt.NDArray[np.float64]:
    """Read a two-column E1290 XRD ``.xy`` file into a NumPy array.

    The file is expected to have one header row followed by rows of
    whitespace-separated ``(2θ, intensity)`` pairs.  Only the first two
    columns are read.

    Parameters
    ----------
    filename : str or Path
        Path to the ``.xy`` file.
    normalize : bool, optional
        When ``True`` (default), divide the intensity column by its
        maximum value so that intensities span [0, 1].

    Returns
    -------
    ndarray of float64
        Array of shape ``(2, N)`` where row 0 contains 2θ values in
        degrees and row 1 contains (optionally normalised) intensities.
    """
    filename = Path(filename)

    xrd = np.loadtxt(filename, skiprows=1, usecols=(0, 1), unpack=True)

    if normalize:
        xrd[1] = xrd[1] / xrd[1].max()

    return xrd.astype(np.float64)
