# PhD Parser

Helper functions to read and process data from different equipment. It's for the Urakawa research group at TU Delft and their laboratories.

The idea is to enable access to experimental data from raw files, and store that data in useful data formats with as much metadata as possible. This repo is not meant to analyze the data using models or for plotting. It simply covers the retrieval, cleaning and processing of data. For deeper analysis and plotting, consult other repos.

![Scientific Data Processing Pipeline](docs/static/data-processing.png)

> [!WARNING]
> This repo is under active development and is changing rapidly.

> [!NOTE]
> As this repo is simply for reading in data and does not contain any scientific output, the use of AI is heavy to speed up the process and curation.

## Intended way of use

If you know how to use git, clone the repo and use it as a package.

Otherwise, simply download the code and copy it.

Each equipment/parser is independently usable.

## Included Equipment

The idea for each equipment is that there is a core `Data` class which utilises diverse parsers to read in and process the raw files from specific equipment and setups. This repo is partially highly specified for our group's equipment. However, parts of it contain parsers for commercial manufacturers and file formats and are hence universally applicable.

---

### Labview

...

---

### Raman

#### Renishaw

1. TXT export

2. WDF export
    - Alex Henderson, DOI:10.5281/zenodo.495477
    - py-wdf-reader (T. Tian, MIT), https://github.com/alchem0x2A/py-wdf-reader
    - SpectroChemPy wdf reader (CeCILL-B)
    - gwyddion renishaw.c

#### B&W Tek

...

---

### XRD

...

---

### XPS

...

---

### TGA

...

---

### Infrared

The `IRData` class is the core container for infrared spectroscopy data. It wraps an `xarray.DataArray` with wavenumbers stored in SI units (m⁻¹) internally and supports both single spectra (1-D) and time-resolved series (2-D, with `scan` and `tos` coordinates). Absolute acquisition timestamps are reconstructed on demand from a `tos_start` stored in metadata plus the elapsed `tos` coordinate — this survives all transformations.

**Constructors**
- `from_arrays` — build from raw numpy arrays (wavenumber in cm⁻¹, values, optional `tos` and `tos_start`)
- `from_xarray` — wrap an existing `xr.DataArray`
- `from_netcdf` — load a previously saved NetCDF file
- `from_omnic_spa` — read Thermo OMNIC `.spa` files (single or series)

**Accessors**
- Unit conversions: `wavenumber`, `wavenumber_per_cm`, `wavelength`, `wavelength_nm`, `frequency`, `energy`, `energy_eV`
- Time: `tos`, `tos_start`, `timestamps`
- Selection: `get_scan`, `get_scan_by_tos`, `get_scan_by_tos_average`, `get_evolution`

**Processing (all immutable — return a new `IRData`)**
- Selection: `sort`, `select_wavenumber_range`, `select_tos_range`
- Smoothing: `smooth_savgol`, `smooth_gaussian`, `smooth_moving`
- Baseline correction: `correct_offset`, `correct_pchip`, `correct_baseline`, `reapply_baseline`
- Averaging: `average_scans`, `average_scans_by_tos`
- Normalisation: `normalise_max`, `normalise_integral`, `normalise_reference`, `normalise_reference_scan`, `normalise_reference_by_tos`, `normalise_value_range`, `normalise_value`
- Arithmetic: `+`, `-` between compatible `IRData` objects

**Export**
- `to_csv` — wavenumber-indexed CSV (cm⁻¹ or m⁻¹)
- `to_netcdf` — round-trippable NetCDF preserving metadata

#### OMNIC (Thermo Scientific)

Low-level parser for `.spa` files in `phd_parser.infrared.omnic`:

- `read_spa` — reads a single `.spa` file, a directory of `.spa` files, or an iterable of paths; returns a dict with stacked `x`, `v` and `tos` arrays plus metadata
- Supports local paths and HTTP(S) URLs
- Time-of-scan (`tos`) derived from: explicit `tos_start`, a fixed `delta_time_seconds` increment, or the embedded file timestamps (default)
- Optional `sort_key` for ordering series (default extracts the "Spectrum Index N" pattern from filenames)
- Extracts core header fields (x/y units, number of points, range) and acquisition datetime

Due to the high overhead of SpectroChemPy [^1], this `read_spa` is a stripped-down version of their parser. For further processing beyond raw file reading, I recommend checking them out.

---

### MS
 
The `MSData` class is the core container for mass spectrometry data. It wraps an `xarray.Dataset` in which each datablock from the source instrument is stored as a separate `DataArray` named `block_{id}` with dims `(time, mz)`. The `time` coordinate is elapsed seconds (SI); optional coordinates `timestamp` (absolute acquisition datetimes) and `cycle` (integer cycle index) ride along on the time axis. When a `tos_start` is provided, time-on-stream is reconstructed on demand as `timestamps - tos_start` — this survives all transformations. Per-block metadata (type, unit, channel definitions) is kept in a separate `block_meta` dict, and a free-form `metadata` dict carries file-level information and an audit trail of any corrections applied.
 
**Constructors**
- `from_arrays` — build from raw numpy arrays (time in seconds, per-block `mz` and `values` dicts, optional `timestamps`, `cycle`, `tos_start`)
- `from_quadstar_asc` — read Pfeiffer Quadstar `.asc` exports (any number of datablocks, auto-assigned m/z columns)
- `from_netcdf` — load a previously saved NetCDF file
**Accessors**
- Blocks: `block_ids`, `n_blocks`, `mz`, `values`, `unit`, `block_type`
- Time: `time`, `n_time`, `timestamps`, `tos`, `tos_start`, `cycle`
- Extraction: `get_trace` (single m/z vs time), `get_traces` (multiple m/z vs time), `get_spectrum` (full m/z at a timepoint)
- Derived: `tic` (total ion current vs time, NaN-safe, cached per block)
**Processing (all immutable — return a new `MSData`)**
- Selection: `select_tos_range`
- Baseline / offset correction: `correct_traces` (shift negative channels up to zero, targeted or across all blocks), `baseline_subtract` (per-channel mean over a tos window)
Both correction methods append an entry to `metadata["trace_corrections"]` so the full processing history is preserved on the object.
 
**Export**
- `to_csv` — time-indexed CSV for a single block (one column per m/z, optional timestamp column)
- `to_netcdf` — round-trippable NetCDF preserving all blocks and coordinates

#### Quadstar for MS in building 67 - Box 5
 
Low-level parser for `.asc` files in `phd_parser.massspec.quadstar`:
 
- `read_export` — reads a Quadstar ASCII export and returns `(meta, df)`: a metadata dict and a tidy `pandas.DataFrame` with one row per cycle
- Parses the four blank-line-delimited sections of the file: file header (name, date, time, converted cycles), cycle/datablock counts, per-datablock channel definitions (mass, min/max, thresholds), and the tabular data
- Builds absolute `Timestamp` column from `Date` + `Time` and localises to a configurable timezone (default `Europe/Amsterdam`)
- Supports arbitrary numbers of datablocks with mixed units (e.g. `A`, `mbar`) and renames channel columns from raw `'b/c'` identifiers to `m{mass}` form
- Builds a `column_map` (original → new name, unit, source datablock) stored on the metadata dict so downstream code (e.g. `MSData.from_quadstar_asc`) can route columns to the correct block
- Optionally drops per-channel `Threshold` columns (`drop_threshold_cols=True` by default at the `MSData` constructor level)

## References

[^1]: Travert, A., & Fernandez, C. (2025). *SpectroChemPy* (Version 0.8.4) [Computer software]. Laboratoire Catalyse and Spectrochemistry (LCS), Normandie Université/CNRS. https://github.com/spectrochempy/spectrochempy (CeCILL-B licence)