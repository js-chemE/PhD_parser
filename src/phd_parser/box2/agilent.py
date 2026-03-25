import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple
import numpy.typing as npt
from ui.renaming import Renaming
import pytz

import logging

LOCAL_TZ_TICKER = "Europe/Amsterdam"
MACHINE_TZ_TICKER = "UTC"

TIMEFORMATS_IN = [
    "%d-%m-%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def count_files_and_folders(directory):
    file_counts = {}
    folder_count = 0

    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isdir(full_path):
            folder_count += 1
        else:
            _, ext = os.path.splitext(entry)
            if ext in file_counts:
                file_counts[ext] += 1
            else:
                file_counts[ext] = 1
    return file_counts, folder_count

def read_estconc(
        filepath: str,
        tos_start: None | pd.Timestamp = None,
        seconds_per_unit: int = 60,
        start_index: int = 10,
        use_local_time: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(filepath, sep='\t', skiprows=2, header=0)
    cols = df.columns
    cols_species = cols[start_index:]

    # Timestamp
    df.drop(columns=["User Name"], inplace=True)
    df["Date"] = df["Date"].astype(str).str.replace('/', '-')
    df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d-%m-%Y %H:%M:%S")
    df.drop(columns=["Date", "Time"], inplace=True)
    df.insert(0, "Timestamp", df.pop("Timestamp"))

    # Set timezone (assume original is UTC)
    df["Timestamp"] = df["Timestamp"].dt.tz_localize(MACHINE_TZ_TICKER)

    # Keep a UTC-free column (naive datetime) — adjusted if toggled
    if use_local_time:
        # Convert to Amsterdam (handles DST automatically)
        #df["Timestamp_machine"] = df["Timestamp"].copy()
        df["Timestamp"] = df["Timestamp"].dt.tz_convert("Europe/Amsterdam")
    else:
        pass

    for col in cols_species:
        df[col] = df[col].astype(str).str.replace('BDL', '')
        df[col] = df[col].astype(str).str.replace('CAL', '')
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)


    if tos_start is not None:
        # Convert tos_start to Timestamp
        tos_start = pd.to_datetime(tos_start, errors='coerce')

        # Ensure tz alignment with Timestamp
        if use_local_time:
            if tos_start.tzinfo is None:
                tos_start = tos_start.tz_localize(LOCAL_TZ_TICKER)
            else:
                tos_start = tos_start.tz_convert(LOCAL_TZ_TICKER)
        else:
            if tos_start.tzinfo is None:
                tos_start = tos_start.tz_localize(MACHINE_TZ_TICKER)
            else:
                tos_start = tos_start.tz_convert(MACHINE_TZ_TICKER)
            
        df["TOS"] = (df["Timestamp"] - tos_start).dt.total_seconds() / seconds_per_unit
    else:
        df["TOS"] = np.nan
    df.insert(1, "TOS", df.pop("TOS"))
    return df, cols_species.to_list()

def read_multiple_estconc(
        directory: str,
        tos_start: None | pd.Timestamp = None,
        seconds_per_unit: int = 60,
        start_index: int = 10,
        use_local_time: bool = True,
        renaming_components: Renaming | Dict[str, str] | None = None) -> Tuple[pd.DataFrame, set]:
    dfs = []
    species = []
    for entry in os.listdir(directory):
        if entry.endswith(".ESTDConc"):
            full_path = os.path.join(directory, entry)
            df, cols_species = read_estconc(
                full_path,
                tos_start,
                seconds_per_unit,
                start_index,
                use_local_time
            )
            dfs.append(df)
            species.extend(cols_species)
    combined = pd.concat(dfs)

    if renaming_components is not None:
        if isinstance(renaming_components, Renaming):
            for col in combined.columns[start_index:]:
                combined.rename(columns={col: renaming_components.get_renaming_single(col)}, inplace=True, errors='ignore')
        elif isinstance(renaming_components, dict):
            combined.rename(columns=renaming_components, inplace=True, errors='ignore')

    return combined, set(species)

def combine_back_and_front(
        front: pd.DataFrame,
        back: pd.DataFrame,
        merge_on: List[str] = ["Timestamp", "TOS", "Sample Id", "File Name", "Method Name"],
        start_index: int = 6,
        prioritize_front: Dict[str, bool] | None = {"CH4" : True}
        ) -> Tuple[pd.DataFrame, List[str]]:
    if prioritize_front is not None:
        for species, front_priority in prioritize_front.items():
            if front_priority:
                back.drop(columns=species, inplace=True)
            else:
                front.drop(columns=species, inplace=True)
    data = pd.merge(left = front, right = back, on=merge_on, how="outer", suffixes=('_front', '_back'))
    species = data.columns[start_index:]
    return data, species.to_list()

def extract_data(
        gc_data: pd.DataFrame,
        species: List[str] = ["Ar", "N2", "CO2", "H2", "MeOH", "MF", "CO", "DME", "CH4", "EtOH"],
        ) -> Tuple[npt.NDArray[np.floating], List[pd.Timestamp], pd.DataFrame]:

    pure_gc_data = pd.DataFrame()
    for s in species:
        try:
            pure_gc_data[s] = gc_data[s].to_numpy()
        except KeyError:
            logger.warning(f"Species {s} not found in the data. Skipping.")
            continue
    gc_data.sort_values(by="Timestamp", inplace=True)
    tos = gc_data["TOS"].to_numpy()
    timestamps = gc_data["Timestamp"].to_list()
    return tos, timestamps, pure_gc_data