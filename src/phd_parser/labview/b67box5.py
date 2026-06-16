import pandas as pd
import os
import numpy as np
from pathlib import Path

from typing import Optional, List, Any, Tuple

import logging

logger = logging.getLogger(__name__)

PRESSURE_CHANNELS = ["Analytic P PV", "Vent P PV"]
TEMPERATURE_CHANNELS = ["Reactor T PV"]
FLOW_CHANNELS = ["F1 He PV", "F1 H2 PV", "F1 CO2 PV", "F2 He PV", "F2 H2 PV", "F2 CO2 PV", "F2 Ar PV"]
PROCESS_CHANNELS = ["timestamp", "Feed"] + ["TOS"]

CHANNEL_META: dict[str, dict[str, Any]] = {
    "Reactor T PV": {
        "unit": "°C",
        "group": "temperature",
        "location": "reactor",
        "kind": "PV",
    },
    "Analytic P PV": {
        "unit": "bar(a)",
        "group": "pressure",
        "location": "analytics",
        "kind": "PV",
    },
    "Vent P PV": {
        "unit": "bar(g)",
        "group": "pressure",
        "location": "vent",
        "kind": "PV",
    },
    "F1 He PV": {
        "unit": "mL/min",
        "group": "flow",
        "controller": "F1",
        "species": "He",
        "kind": "PV",
    },
    "F1 H2 PV": {
        "unit": "mL/min",
        "group": "flow",
        "controller": "F1",
        "species": "H2",
        "kind": "PV",
    },
    "F1 CO2 PV": {
        "unit": "mL/min",
        "group": "flow",
        "controller": "F1",
        "species": "CO2",
        "kind": "PV",
    },
    "F2 He PV": {
        "unit": "mL/min",
        "group": "flow",
        "controller": "F2",
        "species": "He",
        "kind": "PV",
    },
    "F2 H2 PV": {
        "unit": "mL/min",
        "group": "flow",
        "controller": "F2",
        "species": "H2",
        "kind": "PV",
    },
    "F2 CO2 PV": {
        "unit": "mL/min",
        "group": "flow",
        "controller": "F2",
        "species": "CO2",
        "kind": "PV",
    },
    "F2 Ar PV": {
        "unit": "mL/min",
        "group": "flow",
        "controller": "F2",
        "species": "Ar",
        "kind": "PV",
    },
    "Feed": {
        "unit": None,
        "group": "valve",
        "kind": "state",
    },
}


def read(path: str | Path, datetime_format: str = r"%Y-%m-%d_%H-%M-%S", tos_start: Optional[pd.Timestamp] = None, sep: str = "\t", header: Optional[int] = 0, tzinfo: Optional[str] = "Europe/Amsterdam") -> Tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any]]:
    """Read one or more LabView tab-separated log files from building 67, box 5.

    Accepts either a single file path or a directory.  When a directory is
    supplied, all files with ``.txt``, ``.csv``, or ``.ab`` extensions are
    read and concatenated in the order returned by the filesystem.

    The raw dataframe is then passed to :func:`process_log` for timestamp
    parsing, unit conversion, and channel-metadata lookup.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a single log file or a directory containing log files.
    datetime_format : str, optional
        ``strptime`` format string used to parse file-name timestamps
        (default is ``r"%Y-%m-%d_%H-%M-%S"``).  Not applied to the in-file
        timestamp column; see :func:`process_log`.
    tos_start : pandas.Timestamp, optional
        Reference time for the ``tos`` (time-on-stream) coordinate.
        Passed through to :func:`process_log`; defaults to the first
        row's timestamp.
    sep : str, optional
        Column separator for ``pandas.read_csv`` (default is ``"\\t"``).
    header : int or None, optional
        Row number(s) to use as column names, passed to ``pandas.read_csv``
        (default is ``0``).
    tzinfo : str or None, optional
        Timezone name used to localise the parsed timestamps (default is
        ``"Europe/Amsterdam"``).

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe with a ``"timestamp"`` column (tz-aware) and a
        numeric ``"tos"`` column (elapsed seconds).
    dict
        Mapping of channel name to attribute dict (unit, group, etc.).
    dict
        File-level provenance metadata (``"setup"``, ``"n_rows"``,
        ``"tos_start"``, ``"path"``, etc.).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist, or if a file inside a directory cannot be
        found after listing.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_file():
        df = pd.read_csv(path, sep=sep, header=header)

    elif path.is_dir():
        datas = []
        for f in os.listdir(path):
            if f.endswith('.txt') or f.endswith('.csv') or f.endswith('.ab'):

                file_path = path / f

                if not file_path.exists():
                    raise FileNotFoundError(file_path)

                single = pd.read_csv(file_path, sep=sep, header=header)
                datas.append(single)

        df = pd.concat(datas, axis=0)

    cleaned, channel_meta, file_meta = process_log(df, datetime_format=datetime_format, tos_start=tos_start, seconds_per_unit=1.0, tzinfo=tzinfo)

    file_meta["path"] = str(path)

    return cleaned, channel_meta, file_meta

def process_log(
    df: pd.DataFrame, datetime_format: str = r"%Y-%m-%d_%H-%M-%S", tos_start: Optional[pd.Timestamp] = None, seconds_per_unit: float = 1.0, tzinfo: Optional[str] = None
) -> Tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any]]:
    """Clean and annotate a raw LabView log dataframe.

    Performs the following steps in order:

    1. Drops rows with a missing ``"timestamp"`` value (typically the
       trailing blank line written by LabView).
    2. Parses the ``"timestamp"`` column using the fixed format
       ``"%d-%m-%Y %H:%M:%S"``.
    3. Optionally localises timestamps to *tzinfo*.
    4. Computes the ``"tos"`` column as elapsed time since *tos_start*,
       scaled by *seconds_per_unit*.
    5. Looks up per-channel metadata from :data:`CHANNEL_META`; logs a
       warning for unknown channels.
    6. Converts numeric columns that use a comma decimal separator.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataframe as returned by ``pandas.read_csv``.  Must contain a
        ``"timestamp"`` column.
    datetime_format : str, optional
        ``strptime`` format for file-name-derived timestamps (default is
        ``r"%Y-%m-%d_%H-%M-%S"``).  Currently unused inside this function
        but kept for API consistency with :func:`read`.
    tos_start : pandas.Timestamp, optional
        Reference time for the ``"tos"`` coordinate.  Defaults to the
        timestamp of the first (non-null) data row.
    seconds_per_unit : float, optional
        Scale factor applied when computing ``tos`` (default is ``1.0``,
        meaning ``tos`` is in seconds).
    tzinfo : str or None, optional
        Timezone name for localising the parsed timestamps.  Pass ``None``
        to leave them tz-naive (default is ``None``).

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe with ``"timestamp"`` (``datetime64``) and
        ``"tos"`` (``float64``) columns plus all original channel columns
        converted to numeric.
    dict
        Mapping of channel name to attribute dict (unit, group, etc.) drawn
        from :data:`CHANNEL_META`.  Unknown channels receive an empty dict.
    dict
        Provenance metadata: ``"setup"``, ``"n_rows"``,
        ``"filename_timestamp"``, ``"tos_start"``, ``"seconds_per_unit"``.

    Raises
    ------
    ValueError
        If the dataframe does not contain a ``"timestamp"`` column.
    """

    if "timestamp" not in df.columns:
        raise ValueError(
            f"Expected a 'timestamp' column in the data; got {list(df.columns)}"
        )

    # Drop rows with missing timestamps (typically the trailing blank line)
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format="%d-%m-%Y %H:%M:%S", errors="raise"
    )

    if tzinfo is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(tzinfo)


    if tos_start is None:
        tos_start = df["timestamp"].iloc[0]

    df["tos"] = (df["timestamp"] - tos_start).dt.total_seconds() / seconds_per_unit

    channels = [c for c in df.columns if c != "timestamp"]
    channel_meta: dict[str, dict[str, Any]] = {}
    for ch in channels:
        if ch in CHANNEL_META:
            channel_meta[ch] = dict(CHANNEL_META[ch])
            df[ch] = pd.to_numeric(df[ch].str.replace(",", "."), errors="coerce")
        else:
            logger.warning("No metadata for channel %r; leaving empty.", ch)
            channel_meta[ch] = {}

    file_meta: dict[str, Any] = {
        "setup": "b67_box5",
        "n_rows": len(df),
        "filename_timestamp": None,
        "tos_start": tos_start,
        "seconds_per_unit": seconds_per_unit,
    }

    logger.info(f"Processed log with {len(df)} rows and {len(channels)} channels: {channels}")
    return df, channel_meta, file_meta
