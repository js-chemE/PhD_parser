"""What a selection does when the window it was given holds no sample.

Every ``select_tos_range`` / ``select_timestamp_range`` in the package takes
an ``empty`` argument: ``"warn"`` (the default) logs the window and the range
the data cover and returns the empty selection, so a pipeline simply has
nothing left to do; ``"raise"`` turns the same message into a ``ValueError``
for callers that want to stop.
"""

import logging
from typing import Literal

import numpy as np
import numpy.typing as npt

EmptyWindow = Literal["warn", "raise"]

logger = logging.getLogger(__name__)


def window_mask(
    values: npt.NDArray,
    lower: float | None,
    upper: float | None,
    *,
    what: str,
    axis: str,
    unit: str,
    empty: EmptyWindow,
) -> npt.NDArray[np.bool_]:
    """Boolean mask of ``values`` inside ``[lower, upper]`` (inclusive, either bound optional).

    Parameters
    ----------
    values : numpy.ndarray
        The coordinate values the window is cut from.
    lower, upper : float or None
        Inclusive bounds; ``None`` leaves that side open.
    what : str
        What one element is called in the message (``"scan"``, ``"cycle"``, ...).
    axis : str
        Name of the coordinate (``"tos"``, ``"scan id"``).
    unit : str
        Unit of the bounds in the message (``"s"``; empty for a plain index).
    empty : {"warn", "raise"}
        What to do when nothing falls inside the window.

    Returns
    -------
    numpy.ndarray of bool
        The mask; all ``False`` for an empty window under ``"warn"``.

    Raises
    ------
    ValueError
        If the window is empty and ``empty == "raise"``, or if ``empty`` is
        neither value.
    """
    if empty not in ("warn", "raise"):
        raise ValueError(f"empty must be 'warn' or 'raise', got {empty!r}")
    values = np.asarray(values)
    mask = np.ones(values.size, dtype=bool)
    if lower is not None:
        mask &= values >= lower
    if upper is not None:
        mask &= values <= upper
    if not mask.any():
        covered = f"{values.min():g} to {values.max():g} {unit}".rstrip() if values.size else "nothing"
        message = f"No {what} within {axis} [{lower}, {upper}] {unit}: the data cover {covered}".replace("  ", " ")
        if empty == "raise":
            raise ValueError(message)
        logger.warning("%s; returning an empty selection.", message)
    return mask
