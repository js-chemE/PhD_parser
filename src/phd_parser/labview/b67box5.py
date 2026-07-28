import pandas as pd
import os
import numpy as np
from pathlib import Path

from typing import Optional, List, Any, Tuple

import logging

logger = logging.getLogger(__name__)

PRESSURE_CHANNELS = ["Analytic P PV", "Vent P PV"]
TEMPERATURE_CHANNELS = ["Reactor T PV"]
# "F1 CO PV" only exists in logs written from 2026-07 onwards (a CO controller was
# added to line F1); older files simply lack the column.
FLOW_CHANNELS = ["F1 He PV", "F1 H2 PV", "F1 CO2 PV", "F1 CO PV", "F2 He PV", "F2 H2 PV", "F2 CO2 PV", "F2 Ar PV"]
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
    "F1 CO PV": {
        "unit": "mL/min",
        "group": "flow",
        "controller": "F1",
        "species": "CO",
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


def read(path: str | Path, datetime_format: str = r"%Y-%m-%d_%H-%M-%S", tos_start: Optional[pd.Timestamp] = None, sep: str = "\t", header: Optional[int] = 0, tzinfo: Optional[str] = "Europe/Amsterdam", keep_unknown_channels: bool = False) -> Tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any]]:
    """Read one or more LabView tab-separated log files from building 67, box 5.

    Accepts either a single file path or a directory.  When a directory is
    supplied, all files with ``.txt``, ``.csv``, or ``.ab`` extensions are
    read in filename order and concatenated.

    The channel set of the export has changed over time (``"F1 CO PV"`` was
    added in 2026-07), so files of different vintage may carry different
    columns.  Both are read: concatenation takes the union of the columns and
    fills the gaps with NaN, and any column this parser does not recognise is
    skipped rather than allowed to break the read.

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
    keep_unknown_channels : bool, optional
        Read channels missing from :data:`CHANNEL_META` in with empty
        metadata instead of dropping them (default is ``False``).

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
        If *path* does not exist, if a file inside a directory cannot be
        found after listing, or if a directory holds no readable log file.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_file():
        df = pd.read_csv(path, sep=sep, header=header)

    elif path.is_dir():
        datas = []
        columns_per_file: dict[str, list[str]] = {}
        for f in sorted(os.listdir(path)):
            if f.endswith('.txt') or f.endswith('.csv') or f.endswith('.ab'):

                file_path = path / f

                if not file_path.exists():
                    raise FileNotFoundError(file_path)

                single = pd.read_csv(file_path, sep=sep, header=header)
                datas.append(single)
                columns_per_file[f] = list(single.columns)

        if not datas:
            raise FileNotFoundError(
                f"No .txt, .csv or .ab log files found in directory {path}"
            )

        # A directory may mix log formats (e.g. files from before and after the
        # 'F1 CO PV' channel was added). Concatenating takes the union of the
        # columns and leaves NaN where a file did not record a channel.
        all_columns = {c for cols in columns_per_file.values() for c in cols}
        mixed = {
            f: sorted(all_columns - set(cols))
            for f, cols in columns_per_file.items()
            if set(cols) != all_columns
        }
        if mixed:
            logger.warning(
                "Log files in %s do not share the same channels; the missing ones are "
                "filled with NaN: %s",
                path,
                {f: cols for f, cols in mixed.items()},
            )

        df = pd.concat(datas, axis=0, ignore_index=True)

    cleaned, channel_meta, file_meta = process_log(df, datetime_format=datetime_format, tos_start=tos_start, seconds_per_unit=1.0, tzinfo=tzinfo, keep_unknown_channels=keep_unknown_channels)

    file_meta["path"] = str(path)

    return cleaned, channel_meta, file_meta

def process_log(
    df: pd.DataFrame, datetime_format: str = r"%Y-%m-%d_%H-%M-%S", tos_start: Optional[pd.Timestamp] = None, seconds_per_unit: float = 1.0, tzinfo: Optional[str] = None, keep_unknown_channels: bool = False
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
    5. Drops channels that are absent from :data:`CHANNEL_META` (with a
       warning), unless *keep_unknown_channels* is set.  The LabView export
       gains a column now and then — e.g. ``"F1 CO PV"`` from 2026-07 on —
       and an unrecognised one must not break the read.
    6. Looks up per-channel metadata from :data:`CHANNEL_META`.
    7. Converts channel columns that use a comma decimal separator.

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
    keep_unknown_channels : bool, optional
        Read channels missing from :data:`CHANNEL_META` in with empty
        metadata instead of dropping them (default is ``False``).

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe with ``"timestamp"`` (``datetime64``) and
        ``"tos"`` (``float64``) columns plus the recognised channel columns
        converted to numeric.
    dict
        Mapping of channel name to attribute dict (unit, group, etc.) drawn
        from :data:`CHANNEL_META`.  Kept-but-unknown channels receive an
        empty dict.
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

    # 'timestamp' and 'tos' are handled above and are not channels.
    reserved = {"timestamp", "tos"}
    columns = [c for c in df.columns if c not in reserved]

    unknown = [c for c in columns if c not in CHANNEL_META]
    if unknown and not keep_unknown_channels:
        logger.warning(
            "Skipping %d unknown channel(s) %s. The LabView export gained columns this "
            "parser does not know; add them to CHANNEL_META in labview/b67box5.py to "
            "read them in, or pass keep_unknown_channels=True.",
            len(unknown),
            unknown,
        )
        df = df.drop(columns=unknown)
        columns = [c for c in columns if c not in unknown]
    elif unknown:
        logger.warning("No metadata for channel(s) %s; reading them in with empty metadata.", unknown)

    channels = columns
    channel_meta: dict[str, dict[str, Any]] = {}
    for ch in channels:
        # Values use a comma decimal separator; unknown-but-kept channels are
        # converted too, otherwise they stay as strings and break LVData.
        df[ch] = _to_numeric(df[ch])
        channel_meta[ch] = dict(CHANNEL_META.get(ch, {}))

    file_meta: dict[str, Any] = {
        "setup": "b67_box5",
        "n_rows": len(df),
        "filename_timestamp": None,
        "tos_start": tos_start,
        "seconds_per_unit": seconds_per_unit,
    }

    logger.info(f"Processed log with {len(df)} rows and {len(channels)} channels: {channels}")
    return df, channel_meta, file_meta


def _to_numeric(series: pd.Series) -> pd.Series:
    """Convert a raw log column to float, accepting comma or dot as decimal separator."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
