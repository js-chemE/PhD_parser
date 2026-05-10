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

The `LVData` class is the core container for LabView process data. It wraps an `xarray.Dataset` with a single `tos` dimension (elapsed seconds since `tos_start`). Each recorded channel (temperature, pressure, flow, valve state, …) is a separate data variable carrying its own per-channel metadata (`unit`, `group`, `species`, `location`) in `.attrs`. `tos_start` is stored as an ISO-8601 string in `metadata` so it survives all transformations; absolute timestamps are derived on demand from `tos_start + tos` and are never stored as a coordinate.

**Constructors**
- `from_dataframe` — build from a `pandas.DataFrame` with a timestamp column; timezone mismatches between the column and a supplied `tos_start` are reconciled automatically
- `from_netcdf` — load a previously saved NetCDF file
- `from_b67_box5_txt` — read LabView tab-separated exports from the high-pressure setup in building 67, box 5; accepts a single file or a directory of files

**Accessors**
- Core: `channels`, `n_samples`, `sampling_interval`
- Time: `tos`, `tos_start`, `timestamps`
- Channel: `get_channel` (values array by name), `get_channel_unit`, `filter_by_group` (channel names belonging to a group)

**Processing (all immutable — return a new `LVData`)**
- Selection: `select_channels`, `select_group`, `select_tos_range`
- Resampling: `resample` (bin into fixed-width time steps; `mean`, `median`, `first`, or `last` aggregation)
- Smoothing: `smooth_moving` (centered moving average)

**Export**
- `to_dataframe` — `tos`-indexed DataFrame, optional timestamp column
- `to_csv` — `tos`-indexed CSV (timestamp column included when `tos_start` is set)
- `to_netcdf` — round-trippable NetCDF preserving all channels and metadata

#### Building 67, Box 5 (high-pressure setup)

Low-level parser in `phd_parser.labview.b67box5`:

- `read` — reads a single tab-separated file or a whole directory of `.txt`/`.csv` files and concatenates them; returns `(df, channel_meta, file_meta)`
- Timestamps parsed with format `%d-%m-%Y %H:%M:%S` and localised to `Europe/Amsterdam` by default
- Known channels with full metadata: reactor temperature (`Reactor T PV`, °C), analytics pressure (`Analytic P PV`, bar(a)), vent pressure (`Vent P PV`, bar(g)), mass-flow controllers for two lines (`F1`/`F2`) carrying He, H₂, CO₂, and Ar (all mL/min), and a feed valve state (`Feed`)
- Unknown columns are passed through with an empty metadata dict and a warning

---

### Raman

The `RamanData` class is the core container for Raman spectroscopy data. It wraps an `xarray.DataArray` with the Raman shift axis stored in SI units (m⁻¹) internally, using the Stokes convention (positive shift). Three array layouts are supported: `(shift,)` for a single spectrum, `(scan, shift)` for a time series, and `(y, x, shift)` for a spatial map. An optional `tos` coordinate on the `scan` dimension carries elapsed time in seconds. The excitation laser wavelength is stored as a separate `excitation_wavelength_nm` field and used to compute all derived spectral axes as cached properties.

**Constructors**
- `from_arrays` — build from raw numpy arrays (shift in cm⁻¹, values, optional `tos`)
- `from_netcdf` — load a previously saved NetCDF file
- `from_btc655n_export` — read a B&W Tek BTC655N export; converts absolute wavelength to Raman shift using the laser wavelength stored in the file header
- `from_renishaw_txt` — read a Renishaw WiRe ASCII export (two-column: Raman shift in cm⁻¹, intensity)
- `from_renishaw_wdf` — read a Renishaw `.wdf` binary file; supports single spectra, time series, and XY maps; extracts the excitation wavelength directly from the file header

**Accessors**
- Shift axis: `shift` (m⁻¹), `shift_per_cm` (cm⁻¹)
- Derived spectral axes (cached): `excitation_wavenumber`, `excitation_wavenumber_per_cm`, `wavenumber`, `wavenumber_per_cm`, `wavelength`, `wavelength_nm`, `frequency`
- Data: `values`, `n_spectral`, `ndim`, `shape`, `values_label`
- Time: `tos`
- Indexing: `get_scan` (single spectrum by integer scan index), `get_evolution` (intensity vs scan at one or more shifts; nearest or linear interpolation, optional tolerance), `get_map_spectrum` (single spectrum at a spatial position `(x, y)`)

**Processing (all immutable — return a new `RamanData`)**
- `sort` — sort the shift axis ascending or descending
- `select_shift_range` — crop to a spectral window (in cm⁻¹)

**Export**
- `to_csv` — shift-indexed CSV (cm⁻¹ or m⁻¹); columns labelled by `tos_Xs` or `scan_N`; map data (3-D) must use NetCDF
- `to_netcdf` — round-trippable NetCDF preserving all dims, coords, and metadata

#### Renishaw (WiRe)

Low-level parsers in `phd_parser.raman.renishaw`:

- `read_export_txt` — reads a two-column ASCII export (Raman shift in cm⁻¹, intensity); returns `{meta, data}` with instrument and path metadata
- `read_export_wdf` — full binary WDF block-format reader: locates all blocks (`WDF1`, `XLST`, `DATA`, `ORGN`, `WMAP`), extracts the wavenumber axis, spectral data, laser wavenumber, measurement and scan type, stage X/Y/Z positions, per-acquisition timestamps, and map geometry; reshapes data into `(n_spectra, n_points)` for series or `(height, width, n_points)` for maps; returns a `WDFResult` dataclass

The WDF reader is a stripped-down, dependency-free implementation drawing on SpectroChemPy, py-wdf-reader (T. Tian, MIT, DOI:10.5281/zenodo.495477), and gwyddion's renishaw.c.

#### B&W Tek (BTC655N)

Low-level parser in `phd_parser.raman.btc655n`:

- `read_export` — reads a semicolon-delimited `.txt` export; splits the file into a metadata header (semicolon-separated key–value pairs) and a tabular data section starting at the `Pixel` column row; returns `{meta, data}`
- Supported x-keys: `Pixel`, `Wavelength`, `Wavenumber`, `Raman Shift`
- Supported y-keys: `Dark`, `Reference`, `Raw data #1`, `Dark Subtracted #1`, `%TR #1`, `Absorbance #1`, `Irradiance (lumen) #1`
- The `laser_wavelength` metadata field is required by `RamanData.from_btc655n_export` for the wavelength-to-shift conversion

---

### XRD

The `XRDData` class is the core container for X-ray diffraction data. It wraps an `xarray.DataArray` with the 2θ angle stored in degrees throughout (no internal SI conversion). Two layouts are supported: `(angle,)` for a single diffractogram and `(scan, angle)` for a time series, with optional `tos` (seconds) and `timestamp` (datetime) coordinates on the `scan` dimension. Derived angular quantities (θ, radians) are cached properties; d-spacing and the scattering vector Q are computed on demand given a wavelength, defaulting to Cu Kα₁ (1.5406 Å).

**Constructors**
- `from_e1290` — read a two-column `.xy` file from the E1290 setup; optional `normalize` divides intensity by its maximum on load

**Accessors**
- Angle axis: `angle` (2θ, degrees)
- Cached conversions: `angle_rad` (2θ in radians), `theta_deg` (Bragg angle θ in degrees), `theta_rad` (θ in radians)
- Wavelength-dependent (method, not cached): `d_spacing(wavelength_angstrom)` — d-spacing in Å via Bragg's law; `q_vector(wavelength_angstrom)` — scattering vector magnitude Q in Å⁻¹
- Data: `values`, `ndim`, `shape`
- Time: `timestamps`

**Processing (all immutable — return a new `XRDData`)**
- `sort` — sort the angle axis ascending or descending
- `select_angle_range` — crop to a 2θ window `[min_deg, max_deg]`
- `smooth_savgol` — Savitzky–Golay filter (configurable `window_length` and `polyorder`)
- `smooth_gaussian` — Gaussian filter with `sigma_deg` specified in degrees (converted to index units automatically)
- `smooth_moving` — centered moving-average kernel

#### E1290 setup

Low-level parser in `phd_parser.xrd.xrd_e1290`:

- `read_xy_e1290` — reads a two-column whitespace-separated `.xy` file (one header row skipped); returns a `(2, N)` float64 array `[angle, intensity]`; optional `normalize` divides intensity by its maximum

---

### XPS

...

---

### TGA

The `TGAData` class is the core container for thermogravimetric analysis data. It holds parallel `temperature` and `mass` arrays together with an optional `mass_init` for computing normalised mass fractions and an optional `baseline` TGAData stored for provenance after blank subtraction.

**Constructor**
- `from_e2290` — read a Mettler Toledo TGA E2290 export (`.txt`); optional `baseline_path` applies blank subtraction immediately after loading; `in_kelvin=True` (default) converts the temperature axis from °C to K

**Accessors**
- `mass_fraction` — `mass / mass_init`; returns zeros if `mass_init` is not set
- `derivative` — d(mass)/d(temperature) via `numpy.gradient`
- `derivative_fraction` — d(mass_fraction)/d(temperature)

**Processing (all immutable — return a new `TGAData`)**
- Trimming: `cut_front`, `cut_back` — crop by integer index or temperature value
- Baseline correction: `correct` — subtract a blank `TGAData` via interpolation onto the sample's temperature axis; stores the reference as `baseline` on the returned object
- Smoothing: `smooth` — Savitzky–Golay filter (configurable `window_length` and `polyorder`)

#### Mettler Toledo TGA (E2290)

Low-level parser in `phd_parser.tga.e2290`:

- `read_export` — reads a Mettler Toledo TGA `.txt` export and returns a dict with keys: `data` (DataFrame with temperature `Ts` and mass `Value` columns), `units` (list of column units), `curve_name`, `saved_at` and `performed_at` timestamps, `sample_name`, `weight` (initial sample mass in mg), and `method`
- Splits the file into five tagged sections (`Curve Name`, `Curve Values`, `Results`, `Sample`, `Method`) using section-start keywords
- Data rows parsed with whitespace separation; sample weight extracted from the `Sample` section (`<name>, <X.X>mg` format)

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

The `MSData` class is the core container for mass spectrometry data. It wraps an `xarray.Dataset` that holds one `DataArray` per source datablock, named `block_{id}`. The primary block (`block_0`) carries the m/z channels with dims `(cycle, mz)`; auxiliary blocks (pressure, temperature, analog inputs, …) have their own per-block channel dims `(cycle, ch_N)` so that mixed units are never coerced onto a shared "mz" axis. The `cycle` dim (integer scan index) is shared across all blocks, and an optional `tos` coord (time on stream, seconds) rides along it. All metadata lives on the `Dataset` itself: file-level info and the correction audit trail go in `ds.attrs` (including `tos_start` as an ISO-8601 string), while per-block info (unit, type, original channel labels) lives in each DataArray's `.attrs`. When `tos_start` is set, absolute timestamps are reconstructed on demand as `tos_start + tos` — this survives all transformations and round-trips through NetCDF without a side-car dict.

**Constructors**
- `from_arrays` — build from raw numpy arrays (integer `cycle`, per-block `channels` and `values` dicts, optional `tos`, `tos_start`, `block_attrs`, `ds_attrs`)
- `from_quadstar_asc` — read Pfeiffer Quadstar `.asc` exports (any number of datablocks, auto-routes m/z columns into block 0 and everything else into auxiliary blocks)
- `from_netcdf` — load a previously saved NetCDF file

**Accessors**
- Blocks: `block_ids`, `n_blocks`, `channels`, `mz`, `values`, `unit`, `block_type`, `channel_labels`
- Time: `cycle`, `n_cycle`, `tos`, `tos_start`, `timestamps`
- Extraction (m/z block): `get_trace` (single m/z vs cycle; optional centered `rolling_window` mean and `normalize` to [0, 1] or fixed bounds — applied on the returned array without touching the stored data), `get_traces` (multiple m/z vs cycle; delegates to `get_trace` so all parameters pass through), `get_spectrum` (full m/z at a cycle)
- Extraction (any block): `get_channel` (single-channel trace from an auxiliary block)
- Derived: `tic` (total ion current vs cycle, NaN-safe, cached)

**Processing (all immutable — return a new `MSData`)**
- Selection: `select_tos_range`
- Data cleaning: `mask_overloaded` (replace values above a threshold — e.g. detector saturation spikes around 1e38 — with NaN, applied to one block or all)
- Smoothing: `smooth_trace_rolling` (centered rolling mean along the cycle dimension, configurable window and `min_periods`, applied to one block or all)
- Baseline / offset correction: `correct_traces` (shift negative m/z traces up to zero, targeted or across all channels of the m/z block), `baseline_subtract` (per-channel mean over a tos window, applied to one block or all)

All processing methods append an entry to `ds.attrs["trace_corrections"]` so the full processing history is preserved on the object and survives NetCDF round-trips.

**Export**
- `to_csv` — cycle-indexed CSV for a single block (one column per channel, optional `tos_s` and `timestamp` columns)
- `to_netcdf` — round-trippable NetCDF preserving all blocks, coords, and metadata

#### Quadstar for MS in building 67 - Box 5

Low-level parser for `.asc` files in `phd_parser.massspec.quadstar`:

- `read_export` — reads a Quadstar ASCII export and returns `(meta, df)`: a metadata dict and a tidy `pandas.DataFrame` with one row per cycle
- Parses the four blank-line-delimited sections of the file: file header (name, date, time, converted cycles), cycle/datablock counts, per-datablock channel definitions (mass, min/max, thresholds), and the tabular data
- Builds an absolute `Timestamp` column from `Date` + `Time` and localises to a configurable timezone (default `Europe/Amsterdam`)
- Supports arbitrary numbers of datablocks with mixed units (e.g. `A`, `mbar`) and renames m/z channel columns from raw `'b/c'` identifiers to `m{mass}` form
- Builds a `column_map` (original → new name, unit, source datablock) stored on the metadata dict so downstream code (e.g. `MSData.from_quadstar_asc`) can route columns to the correct block — m/z columns go to `block_0`, everything else to `block_N` auxiliary blocks
- Optionally drops per-channel `Threshold` columns (`drop_threshold_cols=True` by default at the `MSData` constructor level)

## References

[^1]: Travert, A., & Fernandez, C. (2025). *SpectroChemPy* (Version 0.8.4) [Computer software]. Laboratoire Catalyse and Spectrochemistry (LCS), Normandie Université/CNRS. https://github.com/spectrochempy/spectrochempy (CeCILL-B licence)