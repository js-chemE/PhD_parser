import pandas as pd
import os
import numpy as np

def read_from_folder(folder_path: str, datetime_format: str = r"%Y-%m-%d_%H-%M-%S", tos_start: pd.Timestamp | None = None, seconds_per_unit: int = 60):
    datas = []
    for f in os.listdir(folder_path):
        if f.endswith('.txt') or f.endswith('.csv') or f.endswith('.ab'):
            file_path = os.path.join(folder_path, f)
            df = pd.read_csv(file_path, sep='\t', header=0)
            datas.append(df)
    combined_df = pd.concat(datas, axis=0)
    try:
        combined_df['Time'] = pd.to_datetime(combined_df['Time'], format=datetime_format)
    except KeyError:
        raise KeyError("Time column not found in the data")
    except ValueError:
        raise ValueError("Time column is not in the correct format")
    except Exception as e:
        raise e
    if tos_start is not None:
        combined_df["TOS"] = (combined_df["Time"] - tos_start).dt.total_seconds() / seconds_per_unit
    else:
        combined_df["TOS"] = np.nan
    
    combined_df.insert(1, "TOS", combined_df.pop("TOS"))
    combined_df.sort_values(by="TOS", inplace=True)
    return combined_df