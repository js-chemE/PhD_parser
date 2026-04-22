import logging
import numpy as np
import numpy.typing as npt
import scipy.constants as const
from typing import TypeAlias
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

Scalar: TypeAlias = float | int
Arraylike: TypeAlias = npt.NDArray
Algebraic: TypeAlias = Scalar | Arraylike


# ================================================
# helper function for transforming x data
# ================================================

def transform_matching_dimensions(values: Algebraic, from_2SI_factor: float = 1.0, to_2SI_factor: float = 1.0) -> Algebraic:
    if np.isclose(from_2SI_factor, to_2SI_factor):
        logger.debug(f"No transformation needed for values with matching 2SI factors: '{from_2SI_factor:8.3e}' and '{to_2SI_factor:8.3e}'. Returning original values.")
        return values
    
    quantity = values * from_2SI_factor
    transformed = quantity / to_2SI_factor
    logger.debug(f"Transformed values from 2SI factor '{from_2SI_factor:8.3e}' to '{to_2SI_factor:8.3e}' using direct scaling. Returning transformed values.")
    return np.asarray(transformed)


def transform_wavenumber_frequency(x: Algebraic, wavenumber_2SI_factor: float = 1.0, frequency_2SI_factor: float = 1.0, forward: bool = True) -> Algebraic:
    if np.isclose(wavenumber_2SI_factor, frequency_2SI_factor):
        logger.debug(f"No transformation needed for wavenumber and frequency with matching 2SI factors: '{wavenumber_2SI_factor:8.3e}' and '{frequency_2SI_factor:8.3e}'. Returning original x data.")
        return x

    if wavenumber_2SI_factor <= 0 or frequency_2SI_factor <= 0:
        logger.error(f"Invalid SI conversion factors: wavenumber_2SI_factor={wavenumber_2SI_factor}, frequency_2SI_factor={frequency_2SI_factor}. Both factors must be positive.")
        raise ValueError(f"Invalid SI conversion factors: wavenumber_2SI_factor={wavenumber_2SI_factor}, frequency_2SI_factor={frequency_2SI_factor}. Both factors must be positive.")
    
    if forward:
        # Convert wavenumber to frequency: e.g f (Hz) = ν (cm^-1) * c (cm/s)
        x_transformed = (x * wavenumber_2SI_factor) * const.c
        return np.asarray(x_transformed / frequency_2SI_factor)
    
    else:
        # Convert frequency to wavenumber: e.g. ν (cm^-1) = f (Hz) / c (cm/s)
        x_transformed = (x * frequency_2SI_factor) / const.c
        return np.asarray(x_transformed / wavenumber_2SI_factor)
    
def transform_wavenumber_energy(x: Algebraic, wavenumber_2SI_factor: float = 1.0, energy_2SI_factor: float = 1.0, forward: bool = True) -> Algebraic:
    if np.isclose(wavenumber_2SI_factor, energy_2SI_factor):
        logger.debug(f"No transformation needed for wavenumber and energy with matching 2SI factors: '{wavenumber_2SI_factor:8.3e}' and '{energy_2SI_factor:8.3e}'. Returning original x data.")
        return x
    if wavenumber_2SI_factor <= 0 or energy_2SI_factor <= 0:
        logger.error(f"Invalid SI conversion factors: wavenumber_2SI_factor={wavenumber_2SI_factor}, energy_2SI_factor={energy_2SI_factor}. Both factors must be positive.")
        raise ValueError(f"Invalid SI conversion factors: wavenumber_2SI_factor={wavenumber_2SI_factor}, energy_2SI_factor={energy_2SI_factor}. Both factors must be positive.")
    
    if forward:
        x_transformed = (x * wavenumber_2SI_factor) * const.h * const.c
        return np.asarray(x_transformed / energy_2SI_factor)
    else:
        x_transformed = (x * energy_2SI_factor) / (const.h * const.c)
        return np.asarray(x_transformed / wavenumber_2SI_factor)