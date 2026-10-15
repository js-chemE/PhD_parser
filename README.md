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

```python
import phd_parser as pp
from pathlib import Path
import pandas as pd
```

The core data containers are available straight from the package, so
`pp.GCData` is the same class as `pp.chromatography.GCData` (likewise
`IRData`, `LVData`, `MSData`, `PhysisorptionData`, `RamanData`, `SpectroscopyData`,
`TGAData`, `XRDData`, and `rename_species` / `SpeciesConvention` from `naming`):

```python
from phd_parser import GCData, LVData, rename_species

gc = GCData.from_ezchrom_rslt(path_rslt)
```

---

## How the package is structured

The package has one job: get data out of instrument files with as little lost
as possible, and hand it over in one consistent form. Everything else
(models, fits, figures) lives in other repos. Four layers do that, and each
one only talks to the one next to it.

```
raw files ──► parser module ──► (vendor converter) ──► core data class ──► your analysis
              "data" + "meta"      common records        GCData, IRData, ...
              dict, as written      SI units              xarray, immutable
```

**1. One subpackage per technique.** `spectroscopy` (with `infrared`, `raman`
and `xps` inside), `massspec`, `labview`, `xrd`, `tga`, `physisorption`,
`chromatography`. Inside it sits the
core class in `core.py` and one parser module per instrument or software
(`omnic.py`, `quadstar.py`, `b67box5.py`). Where a technique has several
manufacturers, each gets its own subfolder (`chromatography/agilent/`), so a
second vendor is added next to the first and never inside it.

**2. Parser modules read, they do not interpret.** A parser returns a plain
dict with a `"data"` and a `"meta"` key, never a data class. It reads
everything the format offers, including what nobody has asked for yet, and
keeps it as the instrument wrote it: vendor names, vendor units, raw counts
next to scaled values. A parser should not fail on a file it can partly read.
Instrument exports drift over time (a new column, a renamed channel), so an
unknown entry is skipped with a warning, and old and new files read side by
side. Time comes from what the instrument recorded, not from a file name.

**3. A vendor converter where several vendors meet.** When different
manufacturers deliver the same kind of data, each vendor module translates
its own dict into one common record format (`ezchrom.to_injections()` for
chromatography). This is the only place where vendor names become generic
ones, minutes become seconds and counts become detector units. The core class
knows the common format only, so adding a vendor never touches it.

**4. One core class per technique.** `GCData`, `IRData`, `MSData`, `LVData`,
... wrap an `xarray` object in a pydantic model, so axes, units and metadata
travel with the numbers. The rules they share:

- **Built by `from_*` classmethods** (`from_ezchrom_rslt`, `from_omnic_spa`, `from_netcdf`), never by hand.
- **Immutable.** Every method returns a new object, so the raw object is always still there to compare against.
- **SI units in storage.** Seconds, m⁻¹, Pa; convenience properties give the usual lab units.
- **One time origin.** `tos` in seconds plus one `tos_start` (see below), so every instrument shares the same clock.
- **Same names for the same operation in every class.** `select_tos_range`, `with_tos_start`, `merge`, `to_netcdf`, `rename_species`. Once one class is known, the others read the same way.
- **Metadata is kept, not trimmed.** What has no typed accessor yet is stored verbatim (JSON-encoded where NetCDF needs it), so it is there when it is needed.
- **Reachable from the root.** `pp.GCData` is `pp.chromatography.GCData`.
- **Every step is recorded.** The object carries its own processing history, and it survives saving (see *Processing history and logging*).
- **One data model where techniques measure the same thing.** IR and Raman are both `SpectroscopyData` on one `wavenumber` axis; `IRData` / `RamanData` only add readers and names.

**Across all of them, two shared helpers.** `naming` maps the spellings of a
species onto one convention, as an explicit `rename_species()` step and never
silently on read. And nothing is ever written next to the raw files: processed
data goes to its own folder (see *Read once, cache as NetCDF*).

---

## Core concepts

### Processing history and logging — every object knows how it was made

This is the feature that makes the rest trustworthy. Two records are kept, and
they answer different questions.

**The processing history: what was done to *this* data.** Every constructor
and every processing step appends one entry to the object it returns, never to
its input:

```python
ir = (
    pp.IRData.from_omnic_spa(dir_spa, tos_start=tos_start)
    .select_tos_range(0, 6 * 3600)
    .correct_baseline(anchor_range_cm=(2600, 2500), control_points_cm=[3450, 2720, 1750])
)
print(ir.format_history())
# 1. 2026-10-15T09:12:03+00:00  from_omnic_spa(filepath='.../spa', ..., tos_start='2025-05-15T10:00:00+02:00', ...)
# 2. 2026-10-15T09:12:04+00:00  select_tos_range(min_s=0, max_s=21600)
# 3. 2026-10-15T09:12:04+00:00  correct_baseline(anchor_range_cm=[2600, 2500], control_points_cm=[3450, 2720, 1750], ...)

ir.history[-1]   # {"step": "correct_baseline", "parameters": {...}, "time": "...", "version": "..."}
```

- One entry per call you made: `correct_baseline` calls `correct_offset` and `correct_pchip` internally, only `correct_baseline` is listed.
- Every argument is stored, files included, so the first entry says which raw data this came from. Arrays longer than 20 values are summarised (shape, first and last value).
- `merge` stores the other segment's own history inside its entry, so a merged run still says how each half was made.
- It lives in `ds.attrs["history"]` and **survives `to_netcdf` / `from_netcdf`**; loading adds a `from_netcdf` entry. A cached file therefore still tells you, months later, which files and which baseline settings produced it.
- Available on `IRData`, `RamanData`, `GCData`, `MSData` and `LVData` (`.history`, `.format_history()`), and `to_spectrochempy()` copies it into the `NDDataset`'s own history. `XRDData`, `TGAData` and `PhysisorptionData` do not record one yet.

The older, step-specific records stay where they were and survive storing too:
`merge_log` (IR), the baseline parameters `baseline_anchor_range_cm` /
`baseline_pchip_*` (used by `reapply_baseline`), `trace_corrections` (MS) and
the `metadata` of `LVData`.

**The log: what the parser decided on the way.** Interpolated a background,
dropped a baseline, found two merge segments overlapping, skipped an unknown
column: these are warnings and info messages on the `phd_parser` logger.

```python
pp.logging.setup_logger("INFO")                               # writes phd_parser.log in the working directory
pp.logging.setup_logger("DEBUG", use_console_handler=True)    # also to the console
pp.logging.disable_logging()
```

The history is attached to the data and travels with it; the log belongs to a
session. Read the log when a result looks odd, read the history to know what a
file is.

---

### `tos_start` — anchoring time-on-stream to the clock

Every data class stores time as *time-on-stream* (`tos`): elapsed seconds since the experiment began. This is a stable, unit-free axis that does not depend on clocks or timezones and survives all transformations and NetCDF round-trips.

To link `tos` back to wall-clock time you supply a `tos_start` once at construction time — a timezone-aware `pandas.Timestamp`. Absolute timestamps are then derived on demand from `tos_start + tos` and are never stored as a coordinate.

```python
tos_start = pd.Timestamp("2026-04-21 14:42:00", tz="Europe/Amsterdam")

lv  = pp.labview.LVData.from_b67_box5_txt(dir_lv, tos_start=tos_start)
ms  = pp.massspec.MSData.from_quadstar_asc(asc_file, tos_start=tos_start)
ir  = pp.infrared.IRData.from_omnic_spa(dir_spa, tos_start=tos_start)
```

Because every instrument shares the same `tos_start`, a single `tos` value means the same moment in all three datasets — making cross-instrument alignment trivial.

#### Changing the origin afterwards

`LVData`, `IRData`, `MSData` and `GCData` each expose the same four immutable methods, mirroring the `with_` / `set_` / `del_` trio used for backgrounds. The distinction that matters is **what stays fixed**:

| Method | What it changes | What stays fixed |
|---|---|---|
| `with_tos_start(tos_start)` | re-anchors: `tos` shifts by minus the distance the origin moved | the absolute timestamps — every sample keeps the wall-clock it was recorded at |
| `set_tos_start(tos_start)` | stamps a new origin, `tos` untouched | the `tos` values — the absolute timestamps all move with the origin |
| `del_tos_start()` | drops the origin | `tos`, which becomes a purely relative axis (`timestamps` → `None`) |
| `move_tos_start_by(delta)` | `with_tos_start(tos_start + delta)` | the absolute timestamps |

Use `with_tos_start` when the *reference point* of the experiment changes — e.g. re-zeroing time to when gas flow started rather than when logging did. Use `set_tos_start` when the elapsed times are right but the wall-clock they were anchored to was wrong.

```python
# Re-zero all three datasets to the moment reaction gas hit the reactor.
t_reaction = pd.Timestamp("2026-04-21 15:12:30", tz="Europe/Amsterdam")
lv, ms, ir = (d.with_tos_start(t_reaction) for d in (lv_raw, ms_raw, ir_raw))
# tos == 0 now means the same instant everywhere; timestamps are unchanged.

# Nudge the origin 90 s later (equivalently: shift every tos down by 90 s).
ir = ir.move_tos_start_by(90)          # seconds
ir = ir.move_tos_start_by("1h30min")   # or any pandas.Timedelta / string
```

Mind the sign: moving the origin **later** makes every `tos` value **smaller**. Pass a negative `delta` to move the origin earlier and grow the `tos` values.

`RamanData` has the same four through `SpectroscopyData`. `XRDData` stores absolute timestamps as a coordinate instead and has no such family.

#### Cutting a window by wall clock

The same four classes let you crop either way: `select_tos_range(min_s, max_s)` in elapsed seconds, or `select_timestamp_range(start, end)` in absolute time. The latter takes a `pandas.Timestamp`, a `datetime`, a `numpy.datetime64` or any string pandas can parse, converts the bounds against `tos_start`, and hands them to `select_tos_range` — so the two always agree. Both bounds are inclusive and either may be `None` for an open end.

```python
# "From when reaction gas was switched in until I went home" — straight off the log book.
ir = ir_raw.select_timestamp_range("2026-04-21 15:12:30", "2026-04-21 18:00:00")

# Same window on all three instruments, no seconds arithmetic in sight.
t0 = pd.Timestamp("2026-04-21 15:12:30", tz="Europe/Amsterdam")
lv, ms, ir = (d.select_timestamp_range(t0, t0 + pd.Timedelta(hours=3))
              for d in (lv_raw, ms_raw, ir_raw))
```

A timezone-naive bound (like the strings above) is read as wall-clock time **in the timezone of `tos_start`**, not as UTC — write the time you read off the log book and it means what you think it means. The method raises if the data carry no `tos_start`, since there is then nothing to resolve absolute times against.

---

### Immutable transformations — every method returns a new object

No processing method mutates the object it is called on. Instead, every method returns a *new* instance with the transformation applied, leaving the original intact. This means you can always go back to the raw data, compare intermediate stages, and chain operations in a single readable expression.

```python
# Each step produces a new object; the raw object is unchanged.
ir = (
    ir_raw
    .select_tos_range(0, 6 * 3600)
    .correct_baseline(anchor_range_cm=(2600, 2500),
                      control_points_cm=[3450, 3100, 2720, 1750, 1230])
)

ms = ms_raw.select_tos_range(0, 6 * 3600).correct_traces().mask_overloaded()
lv = lv_raw.select_tos_range(0, 6 * 3600)
```

The convention `*_raw` for the freshly-loaded object and a plain name for the processed version makes the lineage clear at a glance.

---

### Read once, cache as NetCDF

Parsing hundreds of `.spa` files from disk is slow. The recommended workflow is to parse and pre-process once, save to a compact NetCDF file, then load the NetCDF on every subsequent run. All metadata, backgrounds, coordinates, and processing history survive the round-trip.

```python
# First run — parse raw files, assign background, save.
ir_raw = (
    pp.infrared.IRData.from_omnic_spa(dir_spa, tos_start=tos_start)
    .with_background(pp.infrared.IRData.from_omnic_spa(bg_file, tos_start=tos_start))
)
ir_raw.to_netcdf(cache_path)

# Every subsequent run — fast load.
ir_raw = pp.infrared.IRData.from_netcdf(cache_path)
```

The same pattern applies to `RamanData`, `MSData`, `LVData` and `GCData`.

#### Where the NetCDF file goes: never next to the raw files

`phd_parser` only reads and writes; where a file belongs is decided by the
three data stages of the experiments folder (`21_Experiments/`, described in
its own `README.md` and in the `PhD_execute` README):

| stage | folder | what it is |
|---|---|---|
| raw | `01_raw/<rig>/<run folder>/` | what came from the machine. Read-only: nothing is saved here |
| processed | `02_processed/<project>/<run folder>/<NN_step>/` | anything this package has touched, in numbered steps |
| plot | `03_plot/<project>/<figure>/` | the data behind one figure |

The NetCDF file of a freshly parsed run is the first processed step:
`02_processed/<project>/<run folder>/01_loaded/ir.nc` (and `ms.nc`, `lv.nc`),
with the run folder named as in `01_raw`. `PhD_execute` builds these paths
(`phd_execute.data.paths`) and runs the steps (`phd_execute.drifts.pipeline`).
A cache written into the raw `omnic/` folder is a derived file in the wrong
place.

### Species naming conventions

Every setup writes its species differently: the GC method says `Methanol`, a
mass-flow controller `CH3OH`, a colleague's spreadsheet `MeOH` or `methanol`.
`phd_parser.naming` maps all of them onto one canonical name, so data from
different instruments can be compared, merged and handed to `phd_models`
without a lookup table in every notebook.

A convention is a human-readable JSON file. The standard one,
`src/phd_parser/naming/species/standard.json`, uses the species names of
`phd_models` (`CO2`, `CO`, `H2`, `MeOH`, `H2O`, `CH4`, `DME`, `MF`, `EtOH`,
`PrOH`, `Ar`, `N2`, `He`, ...) as canonical names:

```json
"MeOH": {"name": "methanol", "formula": "CH3OH", "aliases": ["CH4O"]}
```

The key, `name`, `formula` and every alias all resolve to the key. Matching
ignores case, spaces, hyphens, underscores and dots and reads subscript digits
as digits, so `ch4`, `CH₄`, `Methane` and `Methan` are all `CH4`. A
convention in which two species claim the same spelling is refused when it is
loaded.

```python
from phd_parser.naming import rename_species, SpeciesConvention

rename_species(["Methanol", "ch4", "CO₂", "1-propanol"])   # ['MeOH', 'CH4', 'CO2', 'PrOH']

# your own convention: a JSON path, an already read dict, or a SpeciesConvention
rename_species("C1", "path/to/my_convention.json")
rename_species("C1", {"CH4": ["C1", "methane"]})              # flat form, as in phd_ui

# or the standard plus your setup's own spellings
box2 = SpeciesConvention.standard().extend({"CH4": ["C1"]})
```

Names the convention does not know are kept unchanged with a warning
(`unknown="raise"` or `"none"` to change that). Every data class that carries
species names has an immutable `rename_species(convention=None)` that returns
a renamed copy: `GCData` (the species of the peak table and the amounts) and
`LVData` (each channel's `species` attribute). Without a convention it uses
the standard one.

---

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
- Selection: `select_channels`, `select_group`, `select_tos_range`, `select_timestamp_range`
- Time origin: `with_tos_start`, `set_tos_start`, `del_tos_start`, `move_tos_start_by` (see [Changing the origin afterwards](#changing-the-origin-afterwards))
- Resampling: `resample` (bin into fixed-width time steps; `mean`, `median`, `first`, or `last` aggregation)
- Smoothing: `smooth_moving` (centered moving average)
- Naming: `rename_species` (renames each channel's `species` attribute, see [Species naming conventions](#species-naming-conventions))

**Export**
- `to_dataframe` — `tos`-indexed DataFrame, optional timestamp column
- `to_csv` — `tos`-indexed CSV (timestamp column included when `tos_start` is set)
- `to_netcdf` — round-trippable NetCDF preserving all channels and metadata

**Typical workflow**

```python
from pathlib import Path
import pandas as pd
import phd_parser as pp

dir_lv    = Path("path/to/labview")
tos_start = pd.Timestamp("2026-04-21 14:42:00", tz="Europe/Amsterdam")

# --- Load ---
lv_raw = pp.labview.LVData.from_b67_box5_txt(dir_lv, tos_start=tos_start)

# --- Crop to experiment window ---
lv = lv_raw.select_tos_range(0, 6 * 3600)

# --- Access channels ---
T_reactor = lv.get_channel("Reactor T PV")   # returns an xr.DataArray
P_analytic = lv.get_channel("Analytic P PV")

# Axis in hours: lv.tos / 3600
# Absolute timestamps: lv.timestamps

# --- Save / reload ---
lv_raw.to_netcdf(Path("path/to/cache.nc"))
lv_raw = pp.labview.LVData.from_netcdf(Path("path/to/cache.nc"))
```

#### Building 67, Box 5 (high-pressure setup)

Low-level parser in `phd_parser.labview.b67box5`:

- `read` — reads a single tab-separated file or a whole directory of `.txt`/`.csv` files in filename order and concatenates them; returns `(df, channel_meta, file_meta)`
- Timestamps parsed with format `%d-%m-%Y %H:%M:%S` and localised to `Europe/Amsterdam` by default
- Known channels with full metadata: reactor temperature (`Reactor T PV`, °C), analytics pressure (`Analytic P PV`, bar(a)), vent pressure (`Vent P PV`, bar(g)), mass-flow controllers for two lines (`F1` carrying He, H₂, CO₂ and CO; `F2` carrying He, H₂, CO₂ and Ar, all mL/min), and a feed valve state (`Feed`)
- Values use a comma decimal separator and are converted to float on read

**The export changes over time — both old and new files are readable.** The channel set has grown at least once (`F1 CO PV` was added in 2026-07), so files recorded months apart carry different columns:

- A directory holding files from either side of such a change concatenates on the *union* of the columns; a channel a given file did not record is NaN over that file's rows, and a warning names which file was missing which channel
- Columns the parser does not recognise are **skipped** with a warning naming them, rather than breaking the read. Add them to `CHANNEL_META` in `phd_parser.labview.b67box5` to read them in with proper metadata, or pass `keep_unknown_channels=True` to `from_b67_box5_txt` / `read` to keep them with empty metadata

---

### Spectroscopy: one `SpectroscopyData` for infrared and Raman

An IR band and a Raman band of the same vibration sit at the same wavenumber;
the techniques differ in selection rules and in how the signal is recorded,
not in what the axis means (the IR-bands atlas draws both on one axis for the
same reason). So there is one data model, `SpectroscopyData`
(`phd_parser.spectroscopy.core`): `ds["data"]` over `wavenumber` in m⁻¹ (for
Raman, the Raman shift), as `(wavenumber,)`, `(scan, wavenumber)` with `tos`,
or a map `(y, x, wavenumber)`, with an optional `background` and `baseline`.
Every processing step lives there:

- Selection: `select_wavenumber_range`, `select_wavenumber_index_range`, `select_by_idx`, `select_by_tos`, `select_tos_range`, `select_timestamp_range`, `select_scan_id_range`, `sort`
- Time origin: `with_tos_start`, `set_tos_start`, `del_tos_start`, `move_tos_start_by`
- Background and `data_type` conversions (`to_absorbance`, `to_kubelka_munk`, ...), smoothing, baseline correction, averaging, normalisation, signal to noise, Gram-Schmidt, `merge` / `merge_all`, `+` / `-`
- `to_netcdf` / `from_netcdf`, `to_spectrochempy` / `from_spectrochempy`

`IRData` and `RamanData` are thin subclasses. They set `ds.attrs["technique"]`
and the default `data_type` (`single_beam`, `intensity`) and add their file
readers and the names their community uses (`RamanData.shift`,
`select_shift_range`, the excitation wavelength). Everything else is
inherited, so an IR and a Raman spectrum of one sample line up band for band
and plot on one axis. `as_type` moves data between the classes without
touching it (`raman.as_type(IRData)`), and `SpectroscopyData.from_netcdf`
returns the class the file's `technique` names. Merging only rebases data
that is referenced to a background (absorbance, Kubelka-Munk, ...); counts and
single beam merge as they are.

**Handing over to other packages.** `to_spectrochempy()` returns a
SpectroChemPy `NDDataset` (values, wavenumber in cm⁻¹ as `x`, `tos` as `y`
with the timestamps as labels, attributes in `nd.meta`, history in
`nd.history`); `from_spectrochempy(nd)` reads one back, also after processing
there, into the class its `technique` names (`IRData.from_scp` is the strict IR
form of it). `RamanData.to_ramanspy()` returns a RamanSPy `Spectrum`,
`SpectralContainer` (a series) or `SpectralImage` (a map), and
`RamanData.from_ramanspy(container, excitation_wavelength_nm=...)` reads one
back. Both libraries are optional and imported only when called; neither
carries the background or the baseline.

**Where the files are.** `phd_parser/spectroscopy/` holds `core.py` and one
subpackage per technique with its readers: `infrared/`, `raman/` and `xps/`
(a file reader only). The old paths still work and give the same modules:
`pp.infrared` is `pp.spectroscopy.infrared`, and `import phd_parser.infrared.omnic`
still imports.

---

### Raman

The `RamanData` class is the Raman face of `SpectroscopyData` (see [Spectroscopy](#spectroscopy-one-spectroscopydata-for-infrared-and-raman)). The Raman shift is the `wavenumber` axis (m⁻¹, Stokes positive), so `ds["data"]` has dims `(wavenumber,)`, `(scan, wavenumber)` with `tos`, or `(y, x, wavenumber)` for a map, exactly like an IR spectrum. The excitation wavelength is `ds.attrs["excitation_wavelength_nm"]`, what the values are (`intensity`, `counts`, ...) is `data_type`, also readable as `values_label`, and the raw metadata of the source file is the separate field `metadata`. `RamanData(da=..., excitation_wavelength_nm=...)` and NetCDF files of the former DataArray-based class (axis `shift`) still load.

**Constructors**
- `from_arrays` — build from raw numpy arrays (shift in cm⁻¹, values, optional `tos`)
- `from_netcdf` — load a previously saved NetCDF file
- `from_btc655n_export` — read a B&W Tek BTC655N export; converts absolute wavelength to Raman shift using the laser wavelength stored in the file header
- `from_renishaw_txt` — read a Renishaw WiRe ASCII export (two-column: Raman shift in cm⁻¹, intensity)
- `from_renishaw_wdf` — read a Renishaw `.wdf` binary file; supports single spectra, time series, and XY maps; extracts the excitation wavelength directly from the file header

**Accessors**
- Shift axis: `shift` (m⁻¹), `shift_per_cm` (cm⁻¹)
- Shift and scattered light: `wavenumber` / `wavenumber_per_cm` are the shift itself; the scattered light is `scattered_wavenumber`, `scattered_wavenumber_per_cm`, `scattered_wavelength_nm`, `scattered_frequency`, from `excitation_wavenumber`, `excitation_wavenumber_per_cm` (before the merge with IR, `wavenumber` meant the scattered light)
- Data: `values`, `n_spectral`, `ndim`, `shape`, `values_label`
- Time: `tos`, `tos_start`, `timestamps`
- Indexing: `get_scan` (single spectrum by integer scan index), `get_evolution` (intensity vs scan at one or more shifts; nearest or linear interpolation, optional tolerance), `get_map_spectrum` (single spectrum at a spatial position `(x, y)`)

**Processing (all immutable — return a new `RamanData`)**
- `sort` — sort the shift axis ascending or descending
- `select_shift_range` — crop to a spectral window (in cm⁻¹)
- Everything of `SpectroscopyData`: time origin, `select_tos_range`, smoothing, baseline correction, averaging, normalisation, `merge`, signal to noise. On a map the steps that work spectrum by spectrum apply; those that need a scan axis raise

**Export**
- `to_csv` — shift-indexed CSV (cm⁻¹ or m⁻¹); columns labelled by `tos_Xs` or `scan_N`; map data (3-D) must use NetCDF
- `to_netcdf` — round-trippable NetCDF preserving all dims, coords, metadata and the history
- `to_ramanspy` / `from_ramanspy` — RamanSPy `Spectrum` / `SpectralContainer` / `SpectralImage`; `to_spectrochempy` / `from_spectrochempy` — SpectroChemPy `NDDataset`

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

A file reader only for now, without a core class: `phd_parser.spectroscopy.xps.casaxps.read_export` reads a CasaXPS export into the usual `"data"` + `"meta"` dict.

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

The `IRData` class is the core container for infrared spectroscopy data. It wraps an `xarray.Dataset` (`ds`) with wavenumbers stored in SI units (m⁻¹) internally. The primary spectrum is `ds["data"]`; an optional `ds["background"]` (always single-beam, 1-D) is stored alongside it, and an optional `ds["baseline"]` (same shape as `data`) holds the cumulative curve removed by baseline correction. All metadata lives in `ds.attrs`. Two data layouts are supported: `(wavenumber,)` for a single spectrum and `(scan, wavenumber)` for a time series, with an optional `tos` coordinate (seconds) on the `scan` dimension. Absolute acquisition timestamps are reconstructed on demand from `tos_start + tos` and are never stored as a coordinate.

Every spectrum carries a `data_type` attribute that records the physical representation:

| `data_type` | Description |
|---|---|
| `single_beam` | Raw detector signal |
| `absorbance` | −log₁₀(T); default for OMNIC transmission data |
| `transmittance` | I_sample / I_background |
| `reflectance` | I_sample / I_background (reflection experiment) |
| `log_1_r` | −log₁₀(R); pseudo-absorbance for DRIFTS/ATR |
| `kubelka_munk` | (1−R)² / (2R) |

**Constructors**
- `from_arrays` — build from raw numpy arrays (wavenumber in cm⁻¹, values, optional `tos`, `tos_start`, `data_type`)
- `from_xarray` — wrap an existing `xr.DataArray` or `xr.Dataset`
- `from_netcdf` — load a previously saved NetCDF file
- `from_omnic_spa` — read Thermo OMNIC `.spa` files (single file or directory); `data_type` is mapped automatically from the OMNIC header; unknown types raise `ValueError`. `backend` selects the low-level reader: `"auto"` (default) uses SpectroChemPy when installed and falls back to the built-in binary parser, `"spectrochempy"` forces it, `"omnic"` forces the built-in one
- `from_scp` — wrap an existing SpectroChemPy `NDDataset`, however it was produced (read from a file, processed, or built programmatically)

> **One file is just a one-element series.** Reading a single `.spa` gives the same structure as reading a directory — 2-D `(scan, wavenumber)` with one scan and a `tos` coordinate — so `tos_start` applies either way and nothing downstream has to special-case it. A one-scan instance is accepted anywhere a single spectrum is expected, e.g. `series.with_background(background_from_one_file)`.
>
> Every argument after the path is **keyword-only** on `from_omnic_spa` / `from_scp`: `from_omnic_spa(dir, filename)` used to land `filename` in `wavenumber_2SI_factor` and multiply the axis by a string. Join the path yourself — `from_omnic_spa(dir / filename)`. A path that exists but holds no `.spa` file raises `FileNotFoundError` naming the directory, and an unknown `backend` value raises instead of silently falling back to `"auto"`.

**Accessors**
- Spectrum: `values`, `ndim`, `shape`, `data_type`
- Wavenumber axis: `wavenumber` (m⁻¹), `wavenumber_per_cm` (cm⁻¹)
- Cached unit conversions: `wavelength`, `wavelength_nm`, `wavelength_mum`, `frequency`, `energy`, `energy_eV`, `energy_kJ_per_mol`
- Time: `tos`, `tos_start`, `timestamps`
- Background: `has_background`, `background` (numpy array), `background_data_type`
- Baseline: `has_baseline`, `baseline` (numpy array, same shape as `values`), `data_unbaselined` (`values + baseline`, i.e. the spectrum before baseline correction)
- Scan retrieval: `get_scan`, `get_scan_by_tos`, `get_scan_by_tos_average`, `get_evolution`
- Baseline retrieval: `get_baseline_scan`, `get_baseline_by_tos`

**Background management (all immutable — return a new `IRData`)**

The background is always stored as `single_beam`. When a background is already set, switching it triggers automatic recalculation of the data values (e.g. `A_new = A_old + log₁₀(bg_new / bg_old)`).

- `with_background(background, data_type)` — set (no existing background) or switch (recalculates values); accepts `IRData` or numpy array
- `with_background_scan(scan_index)` — use an existing scan as background; converts it from the current `data_type` to `single_beam` automatically
- `with_background_by_tos(target_tos)` — same as above, addressed by time-on-stream
- `set_background(background, data_type)` — force-assign a background without recalculating values (drops any existing background first)
- `del_background()` — remove the background without changing data values

**Baseline management (all immutable — return a new `IRData`)**

`correct_offset`, `correct_pchip`, and `correct_baseline` don't just subtract a curve — they accumulate it into `ds["baseline"]`, so the pre-correction spectrum is never lost.

- `data_unbaselined` — the spectrum as it was before any baseline correction (`values + baseline`)
- `unbaseline()` — returns a new `IRData` with `data` set back to `data_unbaselined` and the baseline cleared, so the regular getters (`get_scan`, `get_scan_by_tos`, `get_evolution`, …) transparently return pre-correction values, e.g. `ir.unbaseline().get_scan_by_tos(3600)`
- `get_baseline_scan(scan_index)` / `get_baseline_by_tos(target_tos)` — fetch the stored baseline curve itself, mirroring `get_scan` / `get_scan_by_tos`
- `del_baseline()` — remove the stored baseline without changing data values

A stored baseline is dropped automatically (with a logger warning) whenever it would no longer be valid: switching `data_type` (`to_absorbance()`, etc.), switching to a different background, or combining two spectra with `+`/`-`. Re-run `correct_baseline()` afterward if you still need it.

**Type conversion (all immutable — return a new `IRData`)**

Conversions follow experiment type: transmission-side (`single_beam → transmittance → absorbance`) and reflection-side (`single_beam → reflectance → log_1_r → kubelka_munk`) are distinct. `absorbance → reflectance` is supported for the common case where OMNIC stores DRIFTS data labelled as absorbance.

- `to_transmittance()` — from `single_beam` (needs background) or `absorbance`
- `to_reflectance()` — from `single_beam` (needs background), `absorbance`, `log_1_r`, or `kubelka_munk`
- `to_absorbance()` — from `single_beam` (needs background) or `transmittance`
- `to_log_1_r()` — from `single_beam` (needs background), `reflectance`, `absorbance`, or `kubelka_munk`
- `to_kubelka_munk()` — from any reflectance-reachable type; routes through `to_reflectance()` internally
- `to_single_beam()` — the inverse of all of the above: re-applies the stored background to get back to raw detector units (needs a background)

**Merging two measurements (all immutable — return a new `IRData`)**

When a run has to be interrupted — the spectrometer is restarted mid-experiment and a *second* background is recorded — the experiment ends up split across two files. `merge` joins them back into one along the scan (time) axis.

Nomenclature used throughout the repo: **merge** joins along the scan/time axis, **extend** would join along the energy axis (wavenumber), and **vstack** would stack 2-D data along a new dimension.

- `merge(other, ...)` — combine two measurements into one
- `IRData.merge_all(items, ...)` — fold `merge` over any number of measurements

The merge itself happens on `single_beam` data, because that is the only representation in which two segments recorded against *different* backgrounds are comparable. You don't have to do that conversion yourself: absorbance (or transmittance, log(1/R), …) segments are converted with `to_single_beam()`, merged, and converted back against the one surviving background — which is exactly right, since it re-references the second segment to the background that survived. Two segments that already share an identical background skip the round trip entirely.

The decisions `merge` makes for you:

| | |
|---|---|
| **data_type** | Preserved. Non-single-beam data is rebased through single beam, so the second segment's *values* change: they are recomputed against the surviving background. The round trip drops any stored baseline, with a warning. Pass `convert_to_single_beam=False` to raise instead and handle it yourself. If a segment carries **no** background (OMNIC often exports finished absorbance and nothing else) there is nothing to rebase with: the scans are merged exactly as recorded and a warning says so. `ds.attrs["merge_log"]` records which of the three happened under `rebasing`. |
| **order** | Segments are ordered by the *absolute* time of their first scan, not by `tos` — two files each starting at `tos=0` are only comparable through their `tos_start`. The earlier one is the *first* segment. |
| **`tos` / `tos_start`** | The merged data keeps the first segment's `tos_start` and leaves its `tos` untouched; the later segment's `tos` is shifted by the difference between the two `tos_start` values, so `timestamps` stays continuous across the join. |
| **background** | Exactly one survives — by default the first segment's, i.e. the one recorded *before* the experiment, not the mid-experiment one. |
| **baseline** | Kept only if *both* segments carry one, otherwise dropped with a warning. |
| **scan ids** | Renumbered `0 … n-1`; a JSON record of the operation is appended to `ds.attrs["merge_log"]`. |

Overrides: `keep_background` (`"first"` / `"last"` / `"none"`, or an explicit array / 1-D `IRData`), `order` (`"auto"` / `"given"`), `sort` (sort merged scans by `tos`; independent of `order`), `tos_offset_seconds` (relate two time axes explicitly — the way to merge segments with no absolute timestamps), `on_overlap` (`"warn"` / `"raise"` / `"ignore"` / `"trim"` the duplicated scans of the second segment), `wavenumber` (`"strict"` requires identical axes, `"interp"` interpolates the second segment onto the first's grid restricted to the common range), and `convert_to_single_beam` (`False` refuses the automatic rebasing).

1-D operands are promoted to a single scan, so two single spectra merge into a 2-D instance and a spectrum can be merged into a series.

**Getting the join right without a hand-recorded restart time.** A restart is an emergency, so nobody writes down when the second measurement began — and you don't need them to. Every `.spa` carries the moment it was recorded, so both segments already sit on the instrument clock. Pass the **same** `tos_start` to both reads (or none at all): each segment's `tos` is then its true offset from that one origin, and `merge` orders by `tos_start + tos`, i.e. by when the scans actually happened.

```python
tos_start = pd.Timestamp("2026-07-23 10:00:00", tz="Europe/Amsterdam")   # one origin, both reads
part_1 = IRData.from_omnic_spa(dir_before_restart, tos_start=tos_start)
part_2 = IRData.from_omnic_spa(dir_after_restart,  tos_start=tos_start)   # no restart time needed

ir_raw = part_1.merge(part_2)
# INFO Joining segments: first ends 2026-07-23 10:23:08+02:00,
#      second starts 2026-07-23 10:43:52+02:00 (gap 1244.0s).
```

`merge` logs where the two segments meet, in wall-clock time, and stores the gap in `ds.attrs["merge_log"]` as `gap_seconds` — check it against what happened in the lab. A negative gap means they overlap, which `on_overlap` handles. Never invent a `tos_start` for the second segment by hand: that replaces a measured time with a guess, and is exactly how the two halves end up on top of each other.

```python
IRData = pp.infrared.IRData

# Before the restart: background recorded at the start of the experiment.
part_1 = (
    IRData.from_omnic_spa(dir_before_restart, tos_start=tos_start_1)
    .with_background(IRData.from_omnic_spa(bg_file_1, tos_start=tos_start_1))
)
# After the restart: a second background, recorded mid-experiment.
part_2 = (
    IRData.from_omnic_spa(dir_after_restart, tos_start=tos_start_2)
    .with_background(IRData.from_omnic_spa(bg_file_2, tos_start=tos_start_2))
)

# One continuous run on one time origin; bg_file_1 survives, bg_file_2 is dropped.
# Works whatever the data_type is: absorbance in, absorbance out, with part_2
# re-referenced to bg_file_1 on the way through.
ir_raw = part_1.merge(part_2)

# Any number of files, in any order — the chronology decides:
ir_raw = IRData.merge_all([part_2, part_1, part_3])
```

**Processing (all immutable — return a new `IRData`; background and baseline are propagated automatically)**
- Selection: `sort`, `select_by_idx`, `select_by_tos`, `select_wavenumber_range`, `select_wavenumber_index_range`, `select_tos_range`, `select_timestamp_range`, `select_scan_id_range`
- Time origin: `with_tos_start`, `set_tos_start`, `del_tos_start`, `move_tos_start_by` (see [Changing the origin afterwards](#changing-the-origin-afterwards))
- Merging: `merge`, `merge_all` (scan axis; see above)
- Smoothing: `smooth_savgol`, `smooth_gaussian`, `smooth_moving`
- Baseline correction: `correct_offset`, `correct_pchip`, `correct_baseline`, `reapply_baseline`, `unbaseline`, `del_baseline`
- Averaging: `average_scans`, `average_scans_by_tos`
- Normalisation: `normalise_max`, `normalise_integral`, `normalise_reference`, `normalise_reference_scan`, `normalise_reference_by_tos`, `normalise_value_range`, `normalise_value`
- Arithmetic: `+`, `-` between compatible `IRData` objects

**Signal quality**
- `snr_windows` — signal-to-noise ratio from separate signal and noise spectral windows; configurable signal metric (`max`, `peak_to_peak`, `integral`, `rms`) and noise metric (`std`, `rms`, `peak_to_peak`)
- `snr_noise_window` — noise estimated from a single flat region (polynomial detrend before estimation)
- `snr_der` — DER-SNR estimator from the second-difference of the spectrum (no reference region needed)
- `snr_repeat` — SNR from scan-to-scan repeatability (requires 2-D data)
- `snr_psd` — power-spectral-density estimator using the high-frequency tail of the FFT

**Gram-Schmidt chromatogram**
- `get_gram_schmidt(reference)` — orthogonal component of each scan relative to a reference subspace; useful for reconstructing a chromatographic profile from a hyphenated IR series
- `get_gram_schmidt_scan(reference_scans)` — convenience wrapper using scan indices as the reference
- `get_gram_schmidt_by_tos(reference_tos)` — convenience wrapper using time-on-stream as the reference

**Export**
- `to_netcdf` — round-trippable NetCDF preserving spectrum, background, coordinates, all metadata and the history
- `to_spectrochempy` — SpectroChemPy `NDDataset` titled with the OMNIC label of `data_type`; `from_spectrochempy` / `IRData.from_scp` read it back

**Typical workflow**

```python
from pathlib import Path
import pandas as pd
import phd_parser as pp

dir_omnic   = Path("path/to/omnic")
dir_reac    = dir_omnic / "02_Reaction"
bg_file     = dir_omnic / "Background_2026-04-21.spa"
cache_path  = Path("path/to/02_processed/<project>/<run folder>/01_loaded/ir.nc")   # not inside the raw folder

tos_start = pd.Timestamp("2026-04-21 14:42:00", tz="Europe/Amsterdam")

# --- First run: parse raw .spa files, assign background, save ---
ir_raw_bg = pp.infrared.IRData.from_omnic_spa(bg_file, tos_start=tos_start)
ir_raw = (
    pp.infrared.IRData.from_omnic_spa(dir_reac, tos_start=tos_start)
    .with_background(ir_raw_bg)           # store single-beam background;
)                                         # data values remain in original data_type
ir_raw.to_netcdf(cache_path)

# --- Subsequent runs: fast load ---
ir_raw = pp.infrared.IRData.from_netcdf(cache_path)

# --- Processing chain ---
anchor_range_cm    = (2600, 2500)
control_points_cm  = [3450, 3100, 2720, 2600, 2500, 2150, 1750, 1230, 800]

ir = (
    ir_raw
    .select_tos_range(0, 6 * 3600)                                   # crop to experiment window
    .correct_baseline(anchor_range_cm, control_points_cm)            # PCHIP baseline correction
    .select_wavenumber_range(1000, 3900)                             # discard noisy edges
)

# --- Inspect the baseline correction itself ---
tos = 1 * 3600
spectrum_uncorrected = ir.unbaseline().get_scan_by_tos(tos)   # spectrum before correction
baseline_curve = ir.get_baseline_by_tos(tos)                  # the curve that was subtracted
# spectrum_uncorrected - baseline_curve == ir.get_scan_by_tos(tos)

# Inspect a single averaged scan at 100 min TOS
spectrum = ir.get_scan_by_tos_average(100 * 60, direction="center", number_of_scans=10)

# Track peak intensity over time
evolution = ir.get_evolution(1600)   # wavenumber in cm⁻¹, returns tos-indexed array

# Average spectra at specific temperature setpoints
ir_temps = ir.average_scans_by_tos(
    tos_targets=[66*60, 126*60, 186*60, 246*60, 306*60],
    time_window=10 * 60,
    direction="backwards",
)
ir_region = ir_temps.select_wavenumber_range(1000, 1750).normalise_value(0.06)

# Signal-to-noise over time
snr = ir.snr_windows(signal_range_cm=(1500, 1750), noise_range_cm=(1800, 1900))
```

#### OMNIC (Thermo Scientific)

Low-level parser for `.spa` files in `phd_parser.infrared.omnic`:

- `read_spa` — reads a single `.spa` file, a directory of `.spa` files, or an iterable of paths; returns a dict with stacked `x`, `v` (always 2-D, one row per file), and `tos` arrays plus metadata (`vlabel`, `vunit`, `xlabel`, `xunit`, datetime list); raises `FileNotFoundError` when the path resolves to no `.spa` file
- `read_spa_datetime` — reads *only* the acquisition timestamp out of a `.spa` (a few bytes, cheap enough to call for thousands of files)
- Supports local paths and HTTP(S) URLs
- Time-of-scan (`tos`) derived from: explicit `tos_start`, a fixed `delta_time_seconds` increment, or the embedded file timestamps (default)

> **Time comes from inside the file, never from the filename.** OMNIC names its exports `Spectrum Index 412 at 0,90 Hours.spa`, but those hours are rounded to two decimals — 0.01 h = **36 s**, so a dozen consecutive scans share one value — and they **restart at zero every time a new series is started**. Both readers therefore take the per-scan acquisition timestamp recorded inside each `.spa`, which is accurate to the second and absolute. The filename hours survive only as a last-resort fallback, with a warning, when no timestamp can be read at all. This is what makes two separately-recorded measurements land on one honest timeline (see [Merging](#merging-two-measurements-all-immutable--return-a-new-irdata)).
- Optional `sort_key` for ordering series (default extracts the "Spectrum Index N" pattern from filenames)
- Extracts core header fields (x/y units, number of points, spectral range) and acquisition datetime
- The `vlabel` from the OMNIC header is mapped to `IRDataType` via `_OMNIC_VLABEL_TO_DATA_TYPE`; unknown labels raise `ValueError` with instructions to extend the mapping

Due to the high overhead of SpectroChemPy [^1], this `read_spa` is a stripped-down version of their parser. For further processing beyond raw file reading, I recommend checking them out.

A SpectroChemPy-backed alternative lives in `phd_parser.infrared.spectrochempy` and is used by default when the library is installed (`backend="auto"`):

- `read_spa` — drop-in replacement for the built-in one: same arguments, same returned dict, so callers never need to know which backend ran
- `read_nddataset` — convert any `NDDataset` (read from a file, processed, or built programmatically) into the same dict; this is what `IRData.from_scp` uses
- `tos` comes from SpectroChemPy's acquisition timestamps when available, otherwise from the hours encoded in the filenames

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
- Selection: `select_tos_range`, `select_timestamp_range`
- Time origin: `with_tos_start`, `set_tos_start`, `del_tos_start`, `move_tos_start_by` (see [Changing the origin afterwards](#changing-the-origin-afterwards))
- Data cleaning: `mask_overloaded` (replace values above a threshold — e.g. detector saturation spikes around 1e38 — with NaN, applied to one block or all)
- Smoothing: `smooth_trace_rolling` (centered rolling mean along the cycle dimension, configurable window and `min_periods`, applied to one block or all)
- Baseline / offset correction: `correct_traces` (shift negative m/z traces up to zero, targeted or across all channels of the m/z block), `baseline_subtract` (per-channel mean over a tos window, applied to one block or all)

All processing methods append an entry to `ds.attrs["trace_corrections"]` so the full processing history is preserved on the object and survives NetCDF round-trips.

**Export**
- `to_csv` — cycle-indexed CSV for a single block (one column per channel, optional `tos_s` and `timestamp` columns)
- `to_netcdf` — round-trippable NetCDF preserving all blocks, coords, and metadata

**Typical workflow**

```python
from pathlib import Path
import pandas as pd
import phd_parser as pp

asc_file  = next(Path("path/to/quadstar").glob("*.asc"))
tos_start = pd.Timestamp("2026-04-21 14:42:00", tz="Europe/Amsterdam")

# --- Load ---
ms_raw = pp.massspec.MSData.from_quadstar_asc(asc_file, tos_start=tos_start)

# --- Processing chain ---
ms = (
    ms_raw
    .select_tos_range(0, 6 * 3600)   # crop to experiment window
    .correct_traces()                 # shift any negative m/z traces up to zero
    .mask_overloaded()                # replace detector-saturation spikes (≈1e38) with NaN
)

# Extract individual m/z traces
trace_28 = ms.get_trace(28)                          # CO / N₂ vs cycle
trace_44 = ms.get_trace(44, normalize=(0, 1))        # CO₂ normalised to [0, 1]

# Save / reload
ms_raw.to_netcdf(Path("path/to/cache.nc"))
ms_raw = pp.massspec.MSData.from_netcdf(Path("path/to/cache.nc"))
```

#### Quadstar for MS in building 67 - Box 5

Low-level parser for `.asc` files in `phd_parser.massspec.quadstar`:

- `read_export` — reads a Quadstar ASCII export and returns `(meta, df)`: a metadata dict and a tidy `pandas.DataFrame` with one row per cycle
- Parses the four blank-line-delimited sections of the file: file header (name, date, time, converted cycles), cycle/datablock counts, per-datablock channel definitions (mass, min/max, thresholds), and the tabular data
- Builds an absolute `Timestamp` column from `Date` + `Time` and localises to a configurable timezone (default `Europe/Amsterdam`)
- Supports arbitrary numbers of datablocks with mixed units (e.g. `A`, `mbar`) and renames m/z channel columns from raw `'b/c'` identifiers to `m{mass}` form
- Builds a `column_map` (original → new name, unit, source datablock) stored on the metadata dict so downstream code (e.g. `MSData.from_quadstar_asc`) can route columns to the correct block — m/z columns go to `block_0`, everything else to `block_N` auxiliary blocks
- Optionally drops per-channel `Threshold` columns (`drop_threshold_cols=True` by default at the `MSData` constructor level)

---

### Physisorption (N₂ physisorption / BET)

The `PhysisorptionData` class is the core, instrument-agnostic container for a single gas-physisorption isotherm *branch* — relative pressure (P/P₀) versus quantity adsorbed (cm³/g STP) — wrapped exactly like a single Raman spectrum or XRD pattern elsewhere in this repo. There is no "branch" concept in the type itself: a reading that produces both adsorption and desorption returns two separate instances. The BET surface-area fit is modelled directly alongside the curve it was derived from (conventionally the adsorption branch), since virtually every physisorption analysis method agrees on it. Other instrument-specific analyses — t-Plot, BJH pore-size distribution, the sample log — are not modelled as typed accessors yet; they are preserved verbatim under `.report` for later use.

**Constructors**
- `from_arrays` — build from raw numpy arrays (`relative_pressure`, `quantity_adsorbed`, optional `bet` dict, optional `attrs`)
- `from_netcdf` — load a previously saved NetCDF file
- `from_tristar_xls` — read a Micromeritics TriStar II 3020 multi-report `.XLS` export; returns a `dict` with `"adsorption"`/`"desorption"` keys (whichever branches were present), only the isotherm and BET results are modelled directly, the rest of the parsed report is stashed under `.report` on each branch

**Accessors**
- `relative_pressure`, `values`, `n_points` — the isotherm curve and its length
- `bet`, `surface_area_bet` — BET fit results (surface area, slope, y-intercept, C constant, monolayer capacity, correlation coefficient, molecular cross-sectional area, each with `*_error`/`*_unit` siblings where applicable) and the headline surface area in m²/g; `None` on a branch with no BET fit
- `report` — raw, instrument-specific report data preserved for provenance (e.g. t-Plot/BJH/sample log for a TriStar export); stored JSON-encoded internally so it survives NetCDF round-trips

**Lookups**
- `get_quantity_adsorbed(target_relative_pressure, method, tolerance)` — quantity adsorbed at one or more relative-pressure values; nearest/linear selection with an optional tolerance check, scalar in → scalar out

**Export**
- `to_netcdf` — round-trippable NetCDF preserving the isotherm, BET results, and the full raw report

**Typical workflow**

```python
from pathlib import Path
import phd_parser as pp

xls_file = Path("path/to/2026-N-198.XLS")

branches = pp.physisorption.PhysisorptionData.from_tristar_xls(xls_file)
adsorption = branches["adsorption"]

print(adsorption.surface_area_bet)                                  # BET surface area, m²/g
q_half = adsorption.get_quantity_adsorbed(0.5)                       # quantity adsorbed at P/P₀ = 0.5

# Everything else MicroActive reported is still there, just not typed yet:
t_plot = adsorption.report["analyses"]["t_plot"]
sample_log = adsorption.report["sample_log"]
```

#### TriStar II 3020 (Micromeritics)

Low-level parser in `phd_parser.physisorption.tristar`:

- `read_export` — parses a MicroActive "print selected reports to Excel" `.XLS` export, where every selected report is laid out as its own block of columns separated by a literal `"|"` divider column, all sharing the same row grid
- A generic state machine handles each block's `key: value` metadata rows, free-text notes, and at most one data table (header row + rows until the next blank row) — including the quirk where a sample-name caption row sits directly above the real column header
- Returns `{"data": {"adsorption": {...}, "desorption": {...}}, "bet": cleaned BET results, "meta": {"header", "summary", "analyses", "sample_log"}}`, where each branch in `"data"` is a `{"relative_pressure", "quantity_adsorbed"}` dict ready to pass into `PhysisorptionData.from_arrays(**branch)`, and `"analyses"` groups each property/analysis (`isotherm`, `bet`, `t_plot`, `bjh_adsorption`, `bjh_desorption`) into its own `{"report", "plots"}` dict

### Chromatography (GC)

The `GCData` class is the vendor-independent container for gas chromatography:
the chromatograms of a series of injections together with their analysis (the
integrated peaks and the amounts per species). It wraps an `xarray.Dataset`:

| what | dims | notes |
|---|---|---|
| one chromatogram variable per detector (`fid`, `tcd`) | `(injection, retention_time_<detector>)` | detector units (`pA`, `25 µV`), retention time in s |
| `peak_*` variables (the peak table) | `(peak,)` | injection, detector, species, retention / start / end time in s, height, area in detector units s |
| `amount`, `area`, `height` | `(injection, species)` | one entry per species and the detector it was quantified on (`CH4` on the FID and on the TCD are two) |

The `injection` dimension is always there, with length 1 for a single
measurement, and `tos` runs along it. Every detector has its own retention
time axis, so instruments that sample their detectors differently fit too.
Injections that differ in run length are padded with `NaN`; injections whose
sampling interval differs (a method changed mid-sequence) are never resampled:
`group_injections` splits them and `from_ezchrom_rslt_groups` returns one
`GCData` per group. The analysis is whatever produced it, named in
`attrs["analysis_source"]` (the vendor software for parsed data).

**Constructors**
- `from_ezchrom_rslt` reads one or more Agilent OpenLab EZChrom `.rslt` folders (e.g. the reduction and the reaction sequence) into one series in time order; `from_ezchrom_rslt_groups` never fails on differing axes
- `from_injections` builds from common injection records, the format every vendor converter produces (`to_injections` gives them back)
- `from_arrays` builds a single chromatogram, `from_netcdf` reloads a saved file

**Accessors**
- `detectors`, `n_injections`, `injection`, `sample_names`, `get_unit`, `get_retention_time`
- Time: `tos`, `tos_start`, `timestamps`
- `get_chromatogram(detector, injection=None)` returns an `xr.DataArray` that slices like a numpy array (`chromatogram[3, 100:200]`) and keeps `tos` and the retention times as coordinates
- `get_peaks()` (long table), `get_amounts(detector=None, quantity="amount")` (wide table, injection x species), `get_evolution(species, detector=None)`

**Processing (all immutable, return a new `GCData`)**
- Selection: `select_by_idx`, `select_tos_range`, `select_timestamp_range` (each keeps the peaks of the injections it keeps), `select_retention_time_range`
- Time origin: `with_tos_start`, `set_tos_start`, `del_tos_start`, `move_tos_start_by` (see [Changing the origin afterwards](#changing-the-origin-afterwards))
- Combining: `merge` appends another `GCData` along the injection axis
- Naming: `rename_species` (see [Species naming conventions](#species-naming-conventions))

**Export**
- `to_netcdf` stores chromatograms that are still raw counts as int32 counts with their scale factor (lossless, half the size); `to_dataframe(detector)` gives one detector as a retention time x injection table

**Typical workflow**

```python
from pathlib import Path
import phd_parser as pp

gc_dir = Path("path/to/2025-05-15_Box2_OP003/GC")
gc_raw = pp.chromatography.GCData.from_ezchrom_rslt(sorted(gc_dir.glob("0*/*.rslt")))
gc = gc_raw.rename_species()                     # Methanol -> MeOH, ...

tcd = gc.get_chromatogram("tcd")                 # (injection, retention time)
co = gc.get_evolution("CO", detector="tcd")      # amount vs injection, with tos
amounts = gc.get_amounts("tcd")                  # injection x species, with tos

gc_raw.to_netcdf(Path("path/to/02_processed/.../01_loaded/gc.nc"))
```

#### Agilent OpenLab CDS EZChrom (Box 2, Agilent 7890B)

Low-level reader in `phd_parser.chromatography.agilent.ezchrom` (also importable as
`phd_parser.chromatography.ezchrom`). It reads the vendor files directly,
without OpenLab or any Agilent library; none of them is encrypted.

- `.dat`, one per injection, is an OLE2 / Compound File Binary container (`CompoundFile` reads it). `read_dat` returns both detector traces as raw ADC counts, their scale factor (1/7680) and unit, and as metadata the full 7890 instrument method XML, instrument setup text, the SQLite audit trail and OLE summary information
- `.acaml` / `.acam_` are ACAML 1.5 XML. `read_acaml` returns every peak with all integrator fields, the species amounts, and the calibration curves with their history down to the calibration `.dat` files
- `.macaml` are the acquisition and data analysis methods as nested name / value / unit sections (`read_method_acaml`); `.rst` and `.cfg` are containers of strings (`read_container_strings`)
- `read_result_set` reads a whole `.rslt` folder, `to_injections` converts it into the common records of `GCData` (SI units, detector units, detectors named from the method: `Front Signal (FID)` becomes `fid`), and `dump_streams` writes every internal stream to disk
- Verified against the AIA netCDF export of the same 120 traces (largest relative deviation 5.8e-8, float32 rounding). Sample k sits at t = k * 0.05 s; the last stored sample is an unacquired zero placeholder, which the netCDF export drops. ACAML heights and areas are in counts and counts s, `to_injections` converts them to detector units

## References

[^1]: Travert, A., & Fernandez, C. (2025). *SpectroChemPy* (Version 0.8.4) [Computer software]. Laboratoire Catalyse and Spectrochemistry (LCS), Normandie Université/CNRS. https://github.com/spectrochempy/spectrochempy (CeCILL-B licence)