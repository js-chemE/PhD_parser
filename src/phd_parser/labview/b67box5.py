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
 