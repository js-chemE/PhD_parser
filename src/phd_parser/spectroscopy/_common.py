"""Constants and small helpers shared by SpectroscopyData and its mixins (no class imports here)."""

from typing import Literal, Optional

import numpy.typing as npt
import xarray as xr

DataType = Literal[
    # absorption / reflection, referenced to a background
    "single_beam", "absorbance", "transmittance", "reflectance", "kubelka_munk", "log_1_r",
    # emission / scattering (Raman)
    "intensity", "counts", "counts_per_second", "arbitrary",
]

BACKGROUND_REFERENCED_TYPES = ("absorbance", "transmittance", "reflectance", "kubelka_munk", "log_1_r")

_OMNIC_VLABEL_TO_DATA_TYPE: dict[str, str] = {
    "absorbance": "absorbance",
    "transmittance": "transmittance",
    "reflectance": "reflectance",
    "single beam": "single_beam",
    "Kubelka_Munk": "kubelka_munk",
    "log(1/R)": "log_1_r",
}

_DATA_TYPE_TO_SCP_TITLE: dict[str, str] = {
    "absorbance": "absorbance",
    "transmittance": "transmittance",
    "reflectance": "reflectance",
    "single_beam": "single beam",
    "kubelka_munk": "Kubelka-Munk",
    "log_1_r": "log(1/R)",
}

_MAP_DIMS = ("y", "x")

_DEFAULT_DATA_TYPE: dict[str, str] = {"infrared": "single_beam", "raman": "intensity"}

_RAMAN_DATA_TYPES = ("intensity", "counts", "counts_per_second", "arbitrary")

def _technique_of_data_type(data_type: Optional[str]) -> Optional[str]:
    """The technique a ``data_type`` belongs to, for a source that names no technique itself."""
    if data_type in _RAMAN_DATA_TYPES:
        return "raman"
    if data_type == "single_beam" or data_type in BACKGROUND_REFERENCED_TYPES:
        return "infrared"
    return None

def _data_type_from_title(title: str) -> Optional[str]:
    """data_type of a spectrochempy / OMNIC title, or None if unknown."""
    title = (title or "").strip()
    if title in _OMNIC_VLABEL_TO_DATA_TYPE:
        return _OMNIC_VLABEL_TO_DATA_TYPE[title]
    lowered = title.lower().replace("-", "_").replace(" ", "_")
    for data_type, scp_title in _DATA_TYPE_TO_SCP_TITLE.items():
        if lowered in (data_type, scp_title.lower().replace("-", "_").replace(" ", "_")):
            return data_type
    return lowered if lowered in DataType.__args__ else None

def _dataset_from_dataarray(da: xr.DataArray) -> xr.Dataset:
    """Turn a NetCDF file of the former DataArray-based Raman class (axis ``shift``) into the current layout."""
    attrs = dict(da.attrs)
    label = attrs.pop("values_label", None) or "intensity"
    attrs.pop("shift_units", None)
    excitation = attrs.get("excitation_wavelength_nm")
    if excitation is not None:
        attrs["excitation_wavelength_nm"] = float(excitation)
    attrs.update({"data_type": label, "technique": "raman", "wavenumber_unit": "m^-1"})
    data = da.rename("data")
    if "shift" in data.dims:
        data = data.rename({"shift": "wavenumber"})
    data.attrs = {}
    return xr.Dataset({"data": data}, attrs=attrs)


def _as_single_spectrum(background) -> npt.NDArray:
    """1-D values of a background SpectroscopyData; a one-scan series counts as a single spectrum."""
    if background.ndim == 1:
        return background.values
    if background.shape[0] == 1:
        return background.values[0]
    raise ValueError(
        f"Background SpectroscopyData must hold a single spectrum, got {background.shape[0]} scans. "
        "Pick one with .select_by_idx(i) / .select_by_tos(t), or average with .average_scans()."
    )


def _is_spectroscopy_data(obj) -> bool:
    """Whether ``obj`` is a SpectroscopyData (imported lazily: the mixins must not import the class)."""
    from phd_parser.spectroscopy.core import SpectroscopyData

    return isinstance(obj, SpectroscopyData)
