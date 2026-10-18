# PhD Parser

Helper functions to read and process data from the instruments of the Urakawa
group at TU Delft. The package gets experimental data out of raw instrument
files with as little lost as possible and hands it over in one consistent form,
with as much metadata as the files offer. It does not model, fit or plot; that
is what `phd_models`, `phd_ui` and `phd_execute` are for.

![Scientific Data Processing Pipeline](docs/static/data-processing.png)

> [!WARNING]
> This repo is under active development and the API still changes.

> [!NOTE]
> The package reads data and produces no scientific result of its own, so AI
> was used heavily to write and curate it. Every reader is checked against
> real files from the lab (the fixtures under `tests/`).

## Installation

With `uv` (recommended, the lockfile pins everything):

```bash
git clone https://github.com/js-chemE/PhD_parser.git
cd PhD_parser
uv sync                     # runtime dependencies only
uv sync --group dev         # plus pytest, the optional backends and ipykernel
```

Into an existing environment:

```bash
pip install -e .                        # core: numpy, pandas, scipy, xarray, pydantic, h5netcdf
pip install -e ".[spectrochempy]"       # SpectroChemPy backend for OMNIC .spa and the NDDataset bridge
pip install -e ".[ramanspy]"            # RamanSPy bridge
pip install -e ".[all]"                 # everything, including reading .spa from URLs
```

Python 3.13 or newer. Run the tests with `pytest`.

```python
import phd_parser as pp
from pathlib import Path
import pandas as pd
```

The core data containers and the spectroscopy readers are available straight
from the package: `pp.GCData` is `pp.chromatography.GCData`, and likewise
`SpectroscopyData`, `IRData`, `RamanData`, `LVData`, `MSData`,
`PhysisorptionData`, `TGAData`, `TPRData`, `XPSData`, `XRDData`,
`rename_species` / `SpeciesConvention` from `naming` and `Pipeline` / `Step`
from `pipeline`.

```python
from phd_parser import GCData, IRData, LVData, rename_species

gc = GCData.from_ezchrom_rslt(path_rslt)
ir = IRData.from_omnic_spa(dir_spa, tos_start=tos_start)   # returns a SpectroscopyData
```

Worked examples per instrument are in [`examples/`](examples/).

---

## How the package is structured

Four layers, and each one only talks to the one next to it.

```
raw files --> parser module --> (vendor converter) --> core data class --> your analysis
              "data" + "meta"     common records        GCData, SpectroscopyData, ...
              dict, as written    SI units              xarray, immutable
```

**1. One subpackage per technique.** `spectroscopy` (with the `infrared`,
`raman` and `xps` readers inside), `massspec`, `labview`, `xrd`, `tga`,
`tpr`, `physisorption`, `chromatography`. Peak shapes are not technique
specific and sit in one top-level module, `phd_parser.kernels` (Gaussian,
Lorentzian, pseudo-Voigt, Voigt; CasaXPS `GL`, `SGL`, `LA`), which every fit
imports. The pseudo-Voigt fraction `eta` is height-weighted unless
`fraction="area"` is passed (the diffraction convention). Inside each subpackage sits the core class in `core.py`
and one parser module per instrument or software (`omnic.py`, `quadstar.py`,
`b67box5.py`). Where a technique has several manufacturers, each gets its own
subfolder (`chromatography/agilent/`), so a second vendor is added next to the
first and never inside it.

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

**4. One core class per technique.** `GCData`, `SpectroscopyData`, `MSData`,
`LVData`, ... wrap an `xarray` object in a pydantic model, so axes, units and
metadata travel with the numbers. The rules they share:

- **Built by `from_*` classmethods** (`from_ezchrom_rslt`, `from_omnic_spa`, `from_netcdf`), never by hand. Everything after the data arguments is keyword-only.
- **Immutable.** Every method returns a new object, so the raw object is always still there to compare against.
- **Shared where it is the same thing.** The time origin (`tos_start`, `timestamps`, the four origin operations, `select_timestamp_range`) and the history accessors are one implementation, mixed into every class (`phd_parser._time`, `phd_parser.history`); `SpectroscopyData` itself is composed of topical mixins (`spectroscopy/_axis.py`, `_selection.py`, `_background.py`, `_baseline.py`, `_signal.py`, `_merge.py`, `_io.py`). See *Mixins* below.
- **SI units in storage.** Seconds, m⁻¹, Pa; convenience properties give the usual lab units.
- **One time origin.** `tos` in seconds plus one `tos_start` (see below), so every instrument shares the same clock.
- **Same names for the same operation in every class.** `select_tos_range`, `with_tos_start`, `merge`, `to_netcdf`, `rename_species`. Once one class is known, the others read the same way.
- **Metadata is kept, not trimmed.** What has no typed accessor yet is stored verbatim (JSON-encoded where NetCDF needs it), so it is there when it is needed.
- **Reachable from the root.** `pp.GCData` is `pp.chromatography.GCData`.
- **Every step is recorded.** The object carries its own processing history, and it survives saving (see *Processing history and logging*).
- **One data class where techniques measure the same thing.** IR and Raman are both `SpectroscopyData` on one `wavenumber` axis; `IRData` and `RamanData` are only the readers.

**Across all of them, two shared helpers.** `naming` maps the spellings of a
species onto one convention, as an explicit `rename_species()` step and never
silently on read. And nothing is ever written next to the raw files: processed
data goes to its own folder (see *Read once, cache as NetCDF*).

---

## Core concepts

### Mixins: one implementation, shared by every class

A *mixin* is a small class that holds a group of related methods and no data
of its own. It is never used on its own: a core class lists it among its base
classes and so gains its methods. Python allows a class to inherit from
several classes at once, so

```python
class LVData(TimeOriginMixin, HistoryMixin, BaseModel): ...
```

is an `LVData` that also has every method of `TimeOriginMixin` (the time
origin) and of `HistoryMixin` (the processing history), on top of what
pydantic's `BaseModel` gives.

**Why mixins and not copies.** Setting a time origin works the same for
LabView, MS, GC and spectroscopy data. Written four times, the four copies
drift apart: one reads a timezone-naive time as UTC, another drops the zone,
and the same experiment ends up with three different clocks. Written once as
a mixin, a fix reaches every class at once, and the method has the same name,
arguments and behaviour everywhere. Learn `with_tos_start` on one class and
you know it on all of them.

**What a mixin asks of its host.** A mixin uses a few attributes that the
class it is mixed into has to provide. That is the contract, and it is short:

| mixin | module | gives | needs from the host | used by |
|---|---|---|---|---|
| `TimeOriginMixin` | `phd_parser._time` | `tos_start`, `timestamps`, `with_tos_start`, `set_tos_start`, `del_tos_start`, `move_tos_start_by`, `select_timestamp_range` | `ds`, `tos`, `_new(ds)`, `select_tos_range` | `SpectroscopyData`, `LVData`, `MSData`, `GCData` |
| `HistoryMixin` | `phd_parser.history` | `history`, `format_history()` | `ds` (the history sits in `ds.attrs["history"]`) | `SpectroscopyData`, `XPSData`, `GCData`, `MSData`, `LVData`, `TPRData` |

A host that keeps the time origin in a second place overrides two hooks
instead of the methods: `LVData` mirrors `tos_start` into its `metadata` by
overriding `_tos_start_raw` and `_with_tos_start`. The methods themselves are
never copied into a class.

**Mixins to split one large class by topic.** `SpectroscopyData` uses the same
mechanism a second way. It has too many methods for one file, so each topic
is its own mixin in its own module, and the class is the sum of them:

| module | topic |
|---|---|
| `spectroscopy/_axis.py` | the axis in every unit, the Raman names, getters |
| `spectroscopy/_selection.py` | cutting by axis, scan, `tos`, scan id |
| `spectroscopy/_background.py` | background and `data_type` conversions |
| `spectroscopy/_baseline.py` | baseline correction |
| `spectroscopy/_signal.py` | smoothing, averaging, normalisation, Gram-Schmidt, SNR |
| `spectroscopy/_merge.py` | `merge` and `merge_all` |
| `spectroscopy/_io.py` | NetCDF, CSV, SpectroChemPy, RamanSPy |

`spectroscopy/core.py` keeps only the fields, the validators, the
constructors and the operators. So to find a method, open the module of its
topic. For a user nothing changes: every method is called on the
`SpectroscopyData` object as usual.

```python
lv.with_tos_start(t_reaction)     # from TimeOriginMixin
ms.with_tos_start(t_reaction)     # the same method, the same rules
ir.correct_baseline(...)          # from _BaselineMixin, called like any method
[c.__name__ for c in type(ir).__mro__]   # every class SpectroscopyData is built from
```

### Processing history and logging: every object knows how it was made

Two records are kept, and they answer different questions.

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

ir.history[-1]   # {"step": "correct_baseline", "parameters": {...}, "time": "...", "version": "0.2.0"}
```

- One entry per call you made: `correct_baseline` calls `correct_offset` and `correct_pchip` internally, only `correct_baseline` is listed.
- Every argument is stored, files included, so the first entry says which raw data this came from. Arrays longer than 20 values are summarised (shape, first and last value).
- `merge` stores the other segment's own history inside its entry, so a merged run still says how each half was made.
- It lives in `ds.attrs["history"]` and **survives `to_netcdf` / `from_netcdf`**; loading adds a `from_netcdf` entry. A cached file therefore still tells you, months later, which files and which baseline settings produced it.
- Available on `SpectroscopyData`, `XPSData`, `GCData`, `MSData`, `LVData` and `TPRData` (`.history`, `.format_history()`), and `to_spectrochempy()` copies it into the `NDDataset`'s own history. `XRDData`, `TGAData` and `PhysisorptionData` do not record one yet.

The older, step-specific records stay where they were and survive storing too:
`merge_log` (spectroscopy), the baseline parameters `baseline_anchor_range_cm` /
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

### `tos_start`: anchoring time-on-stream to the clock

Every data class stores time as *time-on-stream* (`tos`): elapsed seconds since the experiment began. This is a stable, unit-free axis that does not depend on clocks or timezones and survives all transformations and NetCDF round-trips.

To link `tos` back to wall-clock time you supply a `tos_start` once at construction time, a timezone-aware `pandas.Timestamp`. Absolute timestamps are then derived on demand from `tos_start + tos` and are never stored as a coordinate.

```python
tos_start = pd.Timestamp("2026-04-21 14:42:00", tz="Europe/Amsterdam")

lv = pp.LVData.from_b67_box5_txt(dir_lv, tos_start=tos_start)
ms = pp.MSData.from_quadstar_asc(asc_file, tos_start=tos_start)
ir = pp.IRData.from_omnic_spa(dir_spa, tos_start=tos_start)
```

Because every instrument shares the same `tos_start`, a single `tos` value means the same moment in all three datasets, which makes cross-instrument alignment trivial.

#### Changing the origin afterwards

`LVData`, `SpectroscopyData`, `MSData` and `GCData` each expose the same four immutable methods, mirroring the `with_` / `set_` / `del_` trio used for backgrounds. The distinction that matters is **what stays fixed**:

| Method | What it changes | What stays fixed |
|---|---|---|
| `with_tos_start(tos_start)` | re-anchors: `tos` shifts by minus the distance the origin moved | the absolute timestamps: every sample keeps the wall-clock it was recorded at |
| `set_tos_start(tos_start)` | stamps a new origin, `tos` untouched | the `tos` values: the absolute timestamps all move with the origin |
| `del_tos_start()` | drops the origin | `tos`, which becomes a purely relative axis (`timestamps` is then `None`) |
| `move_tos_start_by(delta)` | `with_tos_start(tos_start + delta)` | the absolute timestamps |

Use `with_tos_start` when the *reference point* of the experiment changes, e.g. re-zeroing time to when gas flow started rather than when logging did. Use `set_tos_start` when the elapsed times are right but the wall-clock they were anchored to was wrong.

```python
# Re-zero all three datasets to the moment reaction gas hit the reactor.
t_reaction = pd.Timestamp("2026-04-21 15:12:30", tz="Europe/Amsterdam")
lv, ms, ir = (d.with_tos_start(t_reaction) for d in (lv_raw, ms_raw, ir_raw))
# tos == 0 now means the same instant everywhere; timestamps are unchanged.

# Straight off the log book: a naive time is read in the data's own timezone.
ir = ir.with_tos_start("2026-04-21 15:12:30")

# Nudge the origin 90 s later (equivalently: shift every tos down by 90 s).
ir = ir.move_tos_start_by(90)          # seconds
ir = ir.move_tos_start_by("1h30min")   # or any pandas.Timedelta / string
```

Mind the sign: moving the origin **later** makes every `tos` value **smaller**. Pass a negative `delta` to move the origin earlier and grow the `tos` values. `XRDData` stores absolute timestamps as a coordinate instead and has no such family.

#### Cutting a window by wall clock

The same four classes let you crop either way: `select_tos_range(min_s, max_s)` in elapsed seconds, or `select_timestamp_range(start, end)` in absolute time. The latter takes a `pandas.Timestamp`, a `datetime`, a `numpy.datetime64` or any string pandas can parse, converts the bounds against `tos_start`, and hands them to `select_tos_range`, so the two always agree. Both bounds are inclusive and either may be `None` for an open end. A window that holds no sample logs a warning naming the range the data cover and returns an empty object, so the rest of a pipeline simply has nothing left to do; pass `empty="raise"` to stop with a `ValueError` instead.

```python
# "From when reaction gas was switched in until I went home."
ir = ir_raw.select_timestamp_range("2026-04-21 15:12:30", "2026-04-21 18:00:00")

# Same window on all three instruments, no seconds arithmetic in sight.
t0 = pd.Timestamp("2026-04-21 15:12:30", tz="Europe/Amsterdam")
lv, ms, ir = (d.select_timestamp_range(t0, t0 + pd.Timedelta(hours=3))
              for d in (lv_raw, ms_raw, ir_raw))
```

A timezone-naive bound (like the strings above) is read as wall-clock time **in the timezone of `tos_start`**, not as UTC: write the time you read off the log book and it means what you think it means. The method raises if the data carry no `tos_start`, since there is then nothing to resolve absolute times against.

---

### Immutable transformations: every method returns a new object

No processing method mutates the object it is called on. Instead, every method returns a *new* instance with the transformation applied, leaving the original intact. You can always go back to the raw data, compare intermediate stages, and chain operations in a single readable expression.

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

### Pipelines: write the processing once, run it on every run

A `Pipeline` is a list of steps, each the name of a method and its keyword
arguments. It applies to any data class, maps over a dict or list, is plain
JSON, and can be read off an object's history, so "do to this run what I did
to that one" is one line.

```python
from phd_parser import Pipeline

drifts = Pipeline([
    ("select_tos_range", {"min_s": 0, "max_s": 6 * 3600}),
    ("correct_baseline", {"anchor_range_cm": (2600, 2500), "control_points_cm": [3450, 2720, 1750]}),
    ("select_wavenumber_range", {"min_cm": 1000, "max_cm": 3900}),
    "normalise_max",
], name="drifts standard")

ir = drifts(ir_raw)                                   # one object; every step lands in ir.history
runs = drifts.map({"JV001": ir_1, "JV002": ir_2})     # several, keys kept
regions = Pipeline(["to_cps", ("subtract_background", {"method": "shirley"})]).map(pp.XPSData.from_vamas(path))

drifts.to_json(folder / "pipeline.json")              # next to the processed data
same = Pipeline.from_json(folder / "pipeline.json")
replay = Pipeline.from_history(ir)                    # the steps that made ir, minus the constructor
stage = Pipeline.from_history(ir, since="from_netcdf")  # only what this processing stage did
```

A step whose argument was an array or another data object (`with_background(ir_bg)`) is kept in the history only as a summary, so `from_history` refuses it by name and you add that step by hand. Pipelines are immutable: `append` and `+` return new ones.

### Read once, cache as NetCDF

Parsing hundreds of `.spa` files from disk is slow. Parse and pre-process once, save to a NetCDF file, then load the NetCDF on every subsequent run. All metadata, backgrounds, coordinates and the processing history survive the round trip. Files are NetCDF4 (HDF5), written through `h5netcdf`.

```python
# First run: parse raw files, assign background, save.
ir_raw = (
    pp.IRData.from_omnic_spa(dir_spa, tos_start=tos_start)
    .with_background(pp.IRData.from_omnic_spa(bg_file, tos_start=tos_start))
)
ir_raw.to_netcdf(cache_path)

# Every subsequent run: fast load.
ir_raw = pp.SpectroscopyData.from_netcdf(cache_path)
```

The same pattern applies to `XPSData`, `MSData`, `LVData`, `GCData` and `PhysisorptionData`.

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

---

### Species naming conventions

Every setup writes its species differently: the GC method says `Methanol`, a
mass-flow controller `CH3OH`, a colleague's spreadsheet `MeOH` or `methanol`.
`phd_parser.naming` maps all of them onto one canonical name, so data from
different instruments can be compared, merged and handed to `phd_models`
without a lookup table in every notebook.

A convention is a human-readable JSON file. The standard one,
`src/phd_parser/naming/species/naming_species_standard.json`, uses the species names of
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

## Conventions

The rules the code follows, so that names, units and figures read the same in
every class. They are the user-facing summary of [`CLAUDE.md`](CLAUDE.md),
which holds the full contract and the reasons behind each rule.

### Names

- **Full words, no abbreviations**: `relative_pressure`, not `p_rel`; `quantity_adsorbed`, not `q`; `temperature`, not `temp`. The names of techniques stay as they are (`BET`, `BJH`, `XRD`, `IR`, `MS`, `TGA`), because that is what the methods are called.
- **The unit goes into the name** where it is not obvious: `min_s`, `position_eV`, `sample_weight_mg`, `wavelength_angstrom`, `two_theta_deg`.
- **A chemical species is a *species***: in variables, dimensions, columns and docstrings (`species`, `peak_species`, `rename_species`). Never *component*, which in the lab also means a part of a setup, and not *compound* either, which survives only where a vendor format uses it.
- **Fitted XPS lines are *peaks*** (`peak_name`, `peak_position_eV`), never components.
- **Species names are kept as the instrument wrote them** on read. Renaming is an explicit `rename_species()` step (see *Species naming conventions*).

### Method names

| prefix | meaning | example |
|---|---|---|
| `from_*` | constructor, the only way to build an object; everything after the data arguments is keyword-only | `IRData.from_omnic_spa(path, tos_start=...)` |
| `select_*` | cut out a range; returns a new object | `select_tos_range`, `select_angle_range` |
| `with_*` | set something *and* recalculate what has to change so the physics is preserved | `with_background` recomputes the values, `with_tos_start` keeps the timestamps |
| `set_*` | force-assign, recalculate nothing | `set_tos_start`, `set_background` |
| `del_*` | remove, leave the data alone | `del_background`, `del_tos_start` |
| `get_*` | read something out; no new object, not recorded | `get_trace`, `get_chromatogram` |
| `to_*` | export or convert | `to_netcdf`, `to_dataframe`, `to_single_beam` |

Joining two datasets has three names, never mixed up:

- **extend** joins along the energy-related axis (wavenumber, Raman shift, 2θ, m/z), i.e. widens the spectral range.
- **merge** joins along the *other* axis (scan, time, injection), i.e. appends more scans of the same range (`SpectroscopyData.merge`, `GCData.merge`).
- **vstack** stacks already 2-D data along a new, additional dimension.

A `method` argument that is handed to xarray's `.sel` offers exactly `"nearest"`, `"pad"` and `"backfill"`. Linear interpolation is a different operation (`.interp`), never one of those options.

### Behaviour

- **Immutable.** Every processing method returns a new object; the input is untouched.
- **Recorded.** Every `from_*` and every processing method lands in the object's history (classes listed under *Processing history*).
- **Empty windows warn.** A `select_*_range` that holds no sample logs a warning naming the range of the data and returns an empty object; `empty="raise"` stops instead. It never silently returns everything.
- **Parsers read, they do not interpret.** No normalising, renaming or unit guessing on read. An unknown column is skipped (or kept) with a warning, never a hard failure.
- **Time from the instrument.** `tos` in seconds plus one `tos_start`; times come from what the instrument recorded inside the file, never from a file name. A timezone-naive time is read as wall-clock time in the data's own timezone.
- **Nothing is written next to the raw files.**

### Units and axes

Data is stored in SI units, with convenience properties for the usual lab units.

| class | stored axis | unit |
|---|---|---|
| `SpectroscopyData` | `wavenumber` (for Raman the shift, Stokes positive) | m⁻¹; `axis(unit)` reads any other unit |
| `XPSData` | `binding_energy`, ascending | eV |
| `XRDData` | `angle` (2θ) | degrees |
| `TGAData` | temperature, mass | K, mg |
| `TPRData` | `time`; `temperature` and signals along it | s; °C and mV, as the rig writes them |
| `LVData`, `MSData`, `GCData` | `tos` | s |

### Text and figures

These apply to comments, docstrings, printed output and every figure.

- **No em dashes.** Use a comma, colon, full stop or a new sentence. En dashes in numeric ranges are fine (`780–900 cm⁻¹`).
- **Lowercase** axis labels, tick labels and legend entries; capitals only in titles, proper nouns and where grammar asks for them.
- **Axis labels are `quantity / unit`**: `time / s`, `temperature / K`. A dimensionless quantity gets an en dash: `fraction / –`.
- **Units in power notation**, not slashes: `m^2 s^-1`, `mol m^-3 s^-1`, `mV s g^-1`.
- **Chemistry in mhchem** (`$\ce{CO2}$`) wherever math renders (notes, LaTeX). Code and plot labels stay plain text, since matplotlib has no mhchem.
- **Fonts**: Source Sans 3 for all figure and slide text, Source Serif 4 for print maths and the thesis body; both are bundled in `phd_ui`.

### Code

- NumPy/SciPy-style docstrings on every public class, method, function and property.
- Optional backends (SpectroChemPy, RamanSPy, `requests`) are imported inside the function that needs them, never at module level.
- A new core class is re-exported at the package root in the same change.
- A new reader comes with a small real file under `tests/` and a test that reads it.

---

## Included equipment

Each technique has one core data class that uses one or more parsers to read
and process the raw files of specific instruments. Parts of this are specific
to the group's setups; the readers of commercial file formats (OMNIC, Renishaw
WiRe, Quadstar, EZChrom, TriStar, Mettler Toledo) are usable anywhere.

---

### Spectroscopy: one `SpectroscopyData` for infrared and Raman

An IR band and a Raman band of the same vibration sit at the same wavenumber;
the techniques differ in selection rules and in how the signal is recorded,
not in what the axis means. So there is exactly one data class,
`SpectroscopyData` (`phd_parser.spectroscopy.core`), and no subclasses.
`ds["data"]` runs over `wavenumber` in m⁻¹ (for Raman, the Raman shift, Stokes
positive) as `(wavenumber,)`, `(scan, wavenumber)` with `tos`, or a map
`(y, x, wavenumber)`, with an optional `background` and `baseline`.
`ds.attrs["technique"]` says what recorded the data (`'infrared'`, `'raman'`
or absent); it decides nothing but the default `data_type` and a CSV label.
The raw metadata of the source file is the separate field `metadata`.

**`IRData` and `RamanData` are the readers, not classes of their own.**
`IRData.from_omnic_spa(...)` and `RamanData.from_renishaw_wdf(...)` both
return a `SpectroscopyData`; neither can be instantiated. Use them to create
data and `SpectroscopyData` for everything else, `isinstance` checks and the
generic constructors (`from_netcdf`, `from_arrays`, `from_xarray`,
`from_spectrochempy`, `from_ramanspy`) included.

```python
ir = pp.IRData.from_omnic_spa(dir_spa, tos_start=tos_start)       # technique = 'infrared'
raman = pp.RamanData.from_renishaw_wdf(path_wdf)                    # technique = 'raman'
type(ir) is type(raman) is pp.SpectroscopyData                      # True
difference = ir.select_by_idx(0) - raman                            # one axis, so they combine
```

**Readers**
- `IRData.from_omnic_spa(path, *, tos_start=..., backend="auto")`: Thermo OMNIC `.spa`, one file or a directory. `backend` is `"auto"` (SpectroChemPy when installed, else the built-in parser), `"spectrochempy"` or `"omnic"`. `data_type` comes from the OMNIC header; an unknown label raises
- `RamanData.from_renishaw_wdf(path)`: Renishaw WiRe `.wdf` (single spectra, series with `tos`, maps); excitation wavelength, acquisition time and the header fields come from the file
- `RamanData.from_renishaw_txt(path, excitation_wavelength_nm)`: WiRe two-column text export
- `RamanData.from_btc655n_export(path)`: B&W Tek BTC655N export; wavelength is converted to the shift with the laser wavelength in the file header
- `IRData.from_arrays(wavenumber_per_cm, values, *, tos, tos_start, data_type)` and `RamanData.from_arrays(shift_per_cm, values, *, excitation_wavelength_nm, data_type, tos, tos_start)`: raw arrays with the technique set; `SpectroscopyData.from_arrays(axis, values, axis_unit=..., technique=...)` is the general form and takes the axis in any spectral unit
- `SpectroscopyData.from_netcdf`, `from_xarray`, `from_spectrochempy`, `from_ramanspy`: the generic constructors, on the data class only

> **One file is just a one-element series.** Reading a single `.spa` gives the same structure as reading a directory: 2-D `(scan, wavenumber)` with one scan and a `tos` coordinate, so `tos_start` applies either way and a one-scan instance is accepted anywhere a single spectrum is expected, e.g. `series.with_background(background_from_one_file)`. Every argument after the path is keyword-only, so `from_omnic_spa(dir, filename)` raises instead of multiplying the axis by a string; join the path yourself.

**The axis in any unit.** The axis is stored once, as a wavenumber in m⁻¹
(the Raman shift for Raman), and read in whatever unit the question is asked
in: `axis("um")`, `axis("eV")`, `axis("kJ mol^-1")`, `axis("THz")`, ...
(`axis_label(unit)` gives the matching `quantity / unit` plot label).
`from_arrays(axis, values, axis_unit="nm")` accepts an axis in any of these
units, and `select_axis_range(2.5, 25, "um")` cuts a window in one. The unit
model is that of the IR-bands atlas: wavenumber (`m^-1`, `cm^-1`, ...),
wavelength (`m`, `um`, `nm`, `A`), frequency (`Hz` to `THz`), photon energy
per photon (`J`, `meV`, `eV`, `keV`, `MeV`) and per mole (`J mol^-1`,
`kJ mol^-1`, `cal mol^-1`, `kcal mol^-1`); common spellings (`cm-1`,
`kJ/mol`, `µm`, `Å`) are understood. For a Raman shift the wavenumber,
frequency and energy units read the shift itself (`raman shift / meV`), the
wavelength units read the scattered light (`1 / (excitation - shift)`), which
is what a spectrometer's wavelength axis shows and needs the excitation
wavelength. The full table is `phd_parser.spectroscopy.units`.

**Accessors**
- Axis: `axis(unit)`, `axis_label(unit)`; `wavenumber` (m⁻¹), `wavenumber_per_cm`; Raman names `shift`, `shift_per_cm`; cached shortcuts `wavelength`, `wavelength_nm`, `wavelength_mum`, `frequency`, `energy`, `energy_eV`, `energy_kJ_per_mol`
- Scattered light (needs the excitation wavelength): `excitation_wavelength_nm`, `excitation_wavenumber`, `excitation_wavenumber_per_cm`, `scattered_wavenumber`, `scattered_wavenumber_per_cm`, `scattered_wavelength_nm`, `scattered_frequency`
- Data: `values`, `ndim`, `shape`, `n_spectral`, `data_type`, `values_label`, `da`, `is_map`, `technique`, `metadata`
- Time: `tos`, `tos_start`, `timestamps`
- Background and baseline: `has_background`, `background`, `background_data_type`, `has_baseline`, `baseline`, `data_unbaselined`
- Scans: `get_scan`, `get_scan_by_tos`, `get_scan_by_tos_average`, `get_id_by_tos`, `get_evolution`, `get_map_spectrum`, `get_baseline_scan`, `get_baseline_by_tos`

Every `data_type` records the physical representation:

| `data_type` | Description |
|---|---|
| `single_beam` | raw detector signal |
| `absorbance` | −log₁₀(T); default for OMNIC transmission data |
| `transmittance` | I_sample / I_background |
| `reflectance` | I_sample / I_background (reflection experiment) |
| `log_1_r` | −log₁₀(R); pseudo-absorbance for DRIFTS/ATR |
| `kubelka_munk` | (1−R)² / (2R) |
| `intensity`, `counts`, `counts_per_second`, `arbitrary` | emission and scattering (Raman) |

**Processing (all immutable, all recorded)**
- Selection: `sort`, `select_by_idx`, `select_by_tos`, `select_wavenumber_range`, `select_shift_range`, `select_axis_range` (any unit), `select_wavenumber_index_range`, `select_tos_range`, `select_timestamp_range`, `select_scan_id_range`, `mask_wavenumber_ranges` (NaN inside a window, axis kept), `drop_scans`, `drop_spike_scans` (Hampel test along time, see `find_spike_scans`)
- Time origin: `with_tos_start`, `set_tos_start`, `del_tos_start`, `move_tos_start_by`
- Background: `with_background` (set, or switch and recalculate), `with_background_scan`, `with_background_by_tos`, `set_background` (no recalculation), `del_background`. The background is always stored as `single_beam`
- Conversions: `to_single_beam`, `to_transmittance`, `to_absorbance` (transmission side), `to_reflectance`, `to_log_1_r`, `to_kubelka_munk` (reflection side); `absorbance` to `reflectance` is allowed because OMNIC labels DRIFTS data as absorbance
- Baseline: `correct_offset`, `correct_pchip`, `correct_baseline`, `reapply_baseline`, `correct_drift` (slow drift over the cycles of a modulation experiment), `unbaseline`, `del_baseline`. The subtracted curve accumulates in `ds["baseline"]`, so the pre-correction spectrum is never lost; a stored baseline is dropped, with a warning, when it stops being valid (type conversion, background switch, `+`/`-`)
- Smoothing: `smooth_savgol`, `smooth_gaussian` (sigma in cm⁻¹), `smooth_moving`
- Averaging: `average_scans`, `average_scans_by_tos`
- Normalisation: `normalise_max`, `normalise_integral`, `normalise_reference`, `normalise_reference_scan`, `normalise_reference_by_tos`, `normalise_value_range`, `normalise_value`
- Merging: `merge`, `SpectroscopyData.merge_all` (scan axis, see below)
- Arithmetic: `+`, `-` between compatible objects (the left operand's attributes win)
- Signal quality: `snr_windows`, `snr_noise_window`, `snr_der`, `snr_repeat`, `snr_psd`; Gram-Schmidt: `get_gram_schmidt`, `get_gram_schmidt_scan`, `get_gram_schmidt_by_tos`
- On a map the steps that work spectrum by spectrum apply; those that need a scan axis raise

**Export and bridges**
- `to_netcdf` / `from_netcdf`: round trip of data, background, baseline, coordinates, metadata and history
- `to_csv`: axis-indexed CSV, one column per scan; not for maps
- `to_spectrochempy` / `from_spectrochempy`: SpectroChemPy `NDDataset` (values, wavenumber in cm⁻¹ as `x`, `tos` as `y` with the timestamps as labels, attributes in `nd.meta`, history in `nd.history`)
- `to_ramanspy` / `from_ramanspy`: RamanSPy `Spectrum`, `SpectralContainer` or `SpectralImage`

Both libraries are optional and imported only when called; neither carries the background or the baseline.

#### Merging two measurements

When a run has to be interrupted (the spectrometer is restarted mid-experiment and a *second* background is recorded) the experiment ends up split across two files. `merge` joins them back into one along the scan (time) axis. Nomenclature used throughout the repo: **merge** joins along the scan/time axis, **extend** would join along the energy axis (wavenumber), and **vstack** would stack 2-D data along a new dimension.

The merge happens on `single_beam` data, because that is the only representation in which two segments recorded against *different* backgrounds are comparable. Absorbance (or transmittance, log(1/R), ...) segments are converted with `to_single_beam()`, merged, and converted back against the one surviving background, which re-references the second segment to the background that survived. Two segments that already share a background skip the round trip.

| | |
|---|---|
| **data_type** | Preserved. Background-referenced data is rebased through single beam, so the second segment's values are recomputed against the surviving background. The round trip drops any stored baseline, with a warning. `convert_to_single_beam=False` raises instead. A segment without a background is merged as recorded, with a warning; `ds.attrs["merge_log"]` records which of the three happened under `rebasing`. |
| **order** | Segments are ordered by the *absolute* time of their first scan, not by `tos`. The earlier one is the *first* segment. |
| **`tos` / `tos_start`** | The merged data keeps the first segment's `tos_start`; the later segment's `tos` is shifted by the difference between the two origins, so `timestamps` stays continuous across the join. |
| **background** | Exactly one survives, by default the first segment's, i.e. the one recorded *before* the experiment. |
| **baseline** | Kept only if *both* segments carry one, otherwise dropped with a warning. |
| **scan ids** | Renumbered `0 ... n-1`; a JSON record of the operation (gap, overlap, what was kept) is appended to `ds.attrs["merge_log"]`. |

Overrides: `keep_background` (`"first"` / `"last"` / `"none"`, or an explicit array / 1-D `SpectroscopyData`), `order` (`"auto"` / `"given"`), `sort`, `tos_offset_seconds` (relate two time axes explicitly; the way to merge segments without absolute timestamps), `on_overlap` (`"warn"` / `"raise"` / `"ignore"` / `"trim"`), `wavenumber` (`"strict"` or `"interp"` onto the first segment's grid over the common range) and `convert_to_single_beam`. 1-D operands are promoted to a single scan; maps cannot be merged.

**No hand-recorded restart time is needed.** Every `.spa` carries the moment it was recorded, so both segments already sit on the instrument clock. Pass the **same** `tos_start` to both reads (or none at all), and `merge` orders by `tos_start + tos`, i.e. by when the scans actually happened. Never invent a `tos_start` for the second segment by hand: that replaces a measured time with a guess, and is exactly how the two halves end up on top of each other.

```python
tos_start = pd.Timestamp("2026-07-23 10:00:00", tz="Europe/Amsterdam")   # one origin, both reads
part_1 = pp.IRData.from_omnic_spa(dir_before_restart, tos_start=tos_start)
part_2 = pp.IRData.from_omnic_spa(dir_after_restart, tos_start=tos_start)

ir_raw = part_1.merge(part_2)
# INFO Joining segments: first ends 2026-07-23 10:23:08+02:00,
#      second starts 2026-07-23 10:43:52+02:00 (gap 1244.0s).

# Any number of files, in any order; the chronology decides:
ir_raw = pp.SpectroscopyData.merge_all([part_2, part_1, part_3])
```

#### Typical DRIFTS workflow

```python
from pathlib import Path
import pandas as pd
import phd_parser as pp

dir_omnic = Path("path/to/omnic")
bg_file = dir_omnic / "Background_2026-04-21.spa"
cache_path = Path("path/to/02_processed/<project>/<run folder>/01_loaded/ir.nc")   # not inside the raw folder
tos_start = pd.Timestamp("2026-04-21 14:42:00", tz="Europe/Amsterdam")

# First run: parse the raw .spa files, assign the background, save.
ir_raw = (
    pp.IRData.from_omnic_spa(dir_omnic / "02_Reaction", tos_start=tos_start)
    .with_background(pp.IRData.from_omnic_spa(bg_file, tos_start=tos_start))
)
ir_raw.to_netcdf(cache_path)

# Subsequent runs: fast load.
ir_raw = pp.SpectroscopyData.from_netcdf(cache_path)

# Processing chain.
ir = (
    ir_raw
    .select_tos_range(0, 6 * 3600)
    .correct_baseline((2600, 2500), [3450, 3100, 2720, 2600, 2500, 2150, 1750, 1230, 800])
    .select_wavenumber_range(1000, 3900)
)

# Inspect the baseline correction itself.
spectrum_uncorrected = ir.unbaseline().get_scan_by_tos(3600)
baseline_curve = ir.get_baseline_by_tos(3600)
# spectrum_uncorrected - baseline_curve == ir.get_scan_by_tos(3600)

spectrum = ir.get_scan_by_tos_average(100 * 60, number_of_scans=10)      # averaged scan at 100 min
evolution = ir.get_evolution(1600)                                       # band intensity over time
ir_temps = ir.average_scans_by_tos(target_tos=[66 * 60, 126 * 60, 186 * 60], time_window=10 * 60,
                                   direction="backward")
snr = ir.snr_windows(signal_range_cm=(1500, 1750), noise_range_cm=(1800, 1900))
```

#### The file readers behind the namespaces

`phd_parser.spectroscopy.infrared.omnic`:
- `read_spa`: one `.spa` file, a directory or an iterable of paths; returns `x`, `v` (always 2-D, one row per file), `tos` and the header metadata. Supports local paths and HTTP(S) URLs (needs `requests`)
- `read_spa_datetime`: only the acquisition timestamp of a file (a few bytes, cheap enough for thousands of files)
- `resolve_spa_files`: the files a path names, ordered by `Spectrum Index N` or, when the names carry no unique index, by the recorded acquisition time

> **Time comes from inside the file, never from the filename.** OMNIC names its exports `Spectrum Index 412 at 0,90 Hours.spa`, but those hours are rounded to two decimals (0.01 h = **36 s**) and restart at zero every time a new series is started. Both readers take the per-scan acquisition timestamp recorded inside each `.spa`. The filename hours survive only as a last-resort fallback, with a warning.

`phd_parser.spectroscopy.infrared.spectrochempy` is the SpectroChemPy-backed alternative, used by default when the library is installed: `read_spa` has the same arguments and returned dict, and `read_nddataset` turns any `NDDataset` into that dict. `read_spa` is a stripped-down version of SpectroChemPy's parser [^1], kept because the library's import time is long; for processing beyond reading, SpectroChemPy itself is worth a look.

`phd_parser.spectroscopy.raman.renishaw`: `read_export_wdf` (full binary WDF block reader: `WDF1`, `XLST`, `DATA`, `ORGN`, `WMAP`; returns a `WDFResult` dataclass), `wdf_metadata` (its header as a JSON-safe dict) and `read_export_txt`. The WDF reader is a dependency-free implementation drawing on SpectroChemPy, py-wdf-reader (T. Tian, MIT, DOI:10.5281/zenodo.495477) and gwyddion's renishaw.c.

`phd_parser.spectroscopy.raman.btc655n`: `read_export` for the semicolon-delimited `.txt` export (x-keys `Pixel`, `Wavelength`, `Wavenumber`, `Raman Shift`; y-keys `Dark`, `Reference`, `Raw data #1`, `Dark Subtracted #1`, `%TR #1`, `Absorbance #1`, `Irradiance (lumen) #1`).

`phd_parser.spectroscopy.infrared.modulation`: modulation excitation (MES) helpers on the folded cycles: `fold_cycles`, `average_cycles` (recorded), `demodulate`, `phase_of_maximum`. The steps that return a new spectrum are methods of `SpectroscopyData` and land in its history: `mask_wavenumber_ranges`, `find_spike_scans` / `drop_spike_scans`, `drop_scans`, `correct_drift`.

---

### XPS

`XPSData` holds one region of an X-ray photoelectron measurement (`XPS Survey`, `C1s Scan`, ...): `data` over `binding_energy` in eV, the source energy in the attributes so the kinetic energy the analyser actually scanned is always recoverable (`E_B = hν − E_K`), and whatever else the file carried: the analyser transmission function (VAMAS), a CasaXPS fit (`peaks`, `background`, `envelope`). A file with several regions reads into a `dict` of them, like the two branches of an isotherm. The readers work on the instrument's own files, so a CasaXPS session is not needed to get at the counts.

- **Readers**: `from_vamas(path, region=None)` (ISO 14976 `.vms`, the raw export of Thermo Avantage and what CasaXPS reads; every region or one), `from_avantage_avg(path)` (the Avantage `.avg` text dump of one region; when the file says the transmission function was removed the values are not raw counts and `data_type` is `arbitrary`), `from_casaxps_export(path)` (the tab-separated export of a fit), `from_arrays`, `from_netcdf`
- **Accessors**: `binding_energy`, `kinetic_energy`, `values`, `n_points`, `data_type` (`counts`, `cps`, `arbitrary`), `region`, `sample`, `species_label`, `source_energy_eV`, `dwell_time_s`, `n_scans`, `acquisition_time`, `transmission`, `background`, `peaks` (a table), `envelope`, `integrate(min_eV, max_eV)`
- **Processing (immutable, recorded)**: `select_binding_energy_range`, `sort`, `shift_binding_energy(delta_eV)` and `calibrate_binding_energy(reference_eV, measured_eV)` (charge correction, the cumulative shift stays in the attributes), `to_cps` / `to_counts` (through dwell time and scans), `smooth_savgol`, `normalise_max`, `subtract_background("shirley" | "linear", min_eV, max_eV)` (the subtracted curve is kept as `background`)
- **Export**: `to_netcdf` / `from_netcdf`, `to_dataframe`

```python
regions = pp.XPSData.from_vamas(path_vms)                 # {"C1s Scan": XPSData, "O1s Scan": ..., "XPS Survey": ...}
c1s = (
    regions["C1s Scan"]
    .to_cps()
    .calibrate_binding_energy(284.8, measured_eV=285.3)   # adventitious carbon
    .subtract_background("shirley", min_eV=282, max_eV=292)
)
area = c1s.integrate(282, 292)
```

The parsers are `phd_parser.spectroscopy.xps.vamas.read_vamas` (NORM / REGULAR files; mapping and depth profiles raise `NotImplementedError` rather than reading wrong), `avantage.read_avg` and `casaxps.read_export`.

#### Peak fitting without CasaXPS

`XPSData.fit_peaks` does what CasaXPS did by hand and stores the result the
way a CasaXPS export is stored (`peaks` on top of the background, `envelope`,
the peak coordinates and `attrs["fit"]`), so `peak_table()` reads a fit made
here and an imported one alike. The fit lives in
`phd_parser.spectroscopy.xps.fitting`, the lineshapes in `phd_parser.kernels`:

- **Lineshapes in CasaXPS notation** (`phd_parser.kernels`), parsed by `parse_lineshape`: `GL(m)` (Gaussian-Lorentzian product), `SGL(m)` (sum) and `LA(α, β, n)` (asymmetric Lorentzian convolved with a Gaussian; α acts on the high binding energy side). CasaXPS does not publish how `n` maps onto the Gaussian width, so the mapping used here is our own (`gaussian_width_scale`, 500).
- **`PeakSpec`**: start values, bounds and CasaXPS-style constraints to another peak: `position_offset_from`, `area_ratio_to`, `fwhm_ratio_to` (each fixed, or a `(low, high)` range that is fitted, CasaXPS's `#`), `fwhm_equal_to`.
- **`doublet(...)`** builds both spin-orbit lines at once; `DOUBLETS` holds the nominal splittings and area ratios (Mo 3d 3.13 eV 2:3, Ti 2p 5.7 eV 1:2, Pd 3d 5.26 eV 2:3).
- **`fit_peaks`** on plain arrays: bounded least squares, with standard errors from the Jacobian; `FitResult.bic` decides whether an extra component earns its place.

```python
from phd_parser.spectroscopy.xps import DOUBLETS, doublet

mo3d = regions["Mo3d Scan"].to_cps().calibrate_binding_energy(284.8, measured_eV=285.3)
peaks = [
    *doublet("Mo(VI) 3d", 232.6, lineshape="LA(1.53,243)", **DOUBLETS["Mo 3d"]),
    *doublet("Mo(V) 3d", 231.3, lineshape="LA(1.53,243)", **DOUBLETS["Mo 3d"]),
]
fitted = mo3d.fit_peaks(peaks, background="shirley", min_eV=226, max_eV=240)
fitted.peak_table()      # position, FWHM, area, area fraction and their errors
```

The Mo(V) / Mo(VI) split of a broad, charged spectrum is not determined by the
data alone. Fix the chemical shift and tie the widths (`position_offset_from`,
`fwhm_equal_to`), or the share wanders with the choice of lineshape.

---

### LabView

`LVData` is the core container for LabView process data. It wraps an `xarray.Dataset` with a single `tos` dimension; each recorded channel (temperature, pressure, flow, valve state, ...) is a data variable carrying its own metadata (`unit`, `group`, `species`, `location`) in `.attrs`. `tos_start` is stored as an ISO-8601 string in `metadata`.

- **Constructors**: `from_b67_box5_txt` (a file or a directory of files from the high-pressure setup in building 67, box 5), `from_dataframe`, `from_netcdf`
- **Accessors**: `channels`, `n_samples`, `sampling_interval`, `tos`, `tos_start`, `timestamps`, `get_channel` (a NumPy array), `get_channel_unit`, `filter_by_group`
- **Processing**: `select_channels`, `select_group`, `select_tos_range`, `select_timestamp_range`, the four time-origin methods, `resample` (fixed-width bins; `mean`, `median`, `first`, `last`; NaN samples are skipped), `smooth_moving` (centred window that shrinks at the edges), `rename_species`
- **Export**: `to_dataframe`, `to_csv`, `to_netcdf`

```python
lv_raw = pp.LVData.from_b67_box5_txt(dir_lv, tos_start=tos_start)
lv = lv_raw.select_tos_range(0, 6 * 3600)
temperature = lv.get_channel("Reactor T PV")      # numpy array along lv.tos
```

**Building 67, box 5** (`phd_parser.labview.b67box5.read`): tab-separated exports with a comma decimal separator, timestamps `%d-%m-%Y %H:%M:%S` localised to `Europe/Amsterdam` (the repeated hour of the autumn DST change is resolved from the monotonic log). Known channels carry full metadata: reactor temperature, analytics and vent pressure, the mass-flow controllers of two lines (`F1`: He, H₂, CO₂, CO; `F2`: He, H₂, CO₂, Ar) and the feed valve. **The export changes over time and both old and new files are readable**: a directory holding files from either side of a change concatenates on the *union* of the columns, a channel a file did not record is NaN over its rows, and columns the parser does not know are skipped with a warning naming them. Add them to `CHANNEL_META` to read them in with metadata, or pass `keep_unknown_channels=True`.

---

### MS

`MSData` wraps an `xarray.Dataset` with one `DataArray` per datablock, named `block_{id}`. `block_0` carries the m/z channels with dims `(cycle, mz)`; auxiliary blocks (pressure, temperature, analog inputs) have their own channel dims `(cycle, ch_N)`, so mixed units never share an axis. The `cycle` dim is shared, `tos` rides along it, and `tos_start` sits in `ds.attrs`.

- **Constructors**: `from_quadstar_asc` (Pfeiffer Quadstar `.asc`, one file or a directory; a naive `tos_start` is read in the file's timezone), `from_arrays`, `from_netcdf`
- **Accessors**: `block_ids`, `n_blocks`, `channels`, `mz`, `values`, `unit`, `block_type`, `channel_labels`, `cycle`, `n_cycle`, `tos`, `tos_start`, `timestamps`; `get_trace` (one m/z vs cycle, with optional centred `rolling_window` applied *before* `normalize`), `get_traces`, `get_spectrum`, `get_channel`, `tic`
- **Processing**: `select_tos_range`, `select_timestamp_range`, the four time-origin methods, `mask_overloaded` (detector saturation spikes around 1e38 become NaN), `smooth_trace_rolling`, `correct_traces` (shift negative traces up to zero), `baseline_subtract`. Every step is also logged in `ds.attrs["trace_corrections"]`
- **Export**: `to_csv` (one block), `to_netcdf`

```python
ms_raw = pp.MSData.from_quadstar_asc(asc_file, tos_start=tos_start)
ms = ms_raw.select_tos_range(0, 6 * 3600).correct_traces().mask_overloaded()
trace_44 = ms.get_trace(44, rolling_window=5, normalize=True)
```

**Quadstar** (`phd_parser.massspec.quadstar.read_export`): parses the four blank-line-delimited sections of the export (header, cycle and datablock counts, per-datablock channel definitions, data), builds a timezone-aware `Timestamp` column, renames m/z channels to `m{mass}` and keeps a `column_map` so `MSData` can route every column to its block. Threshold columns are dropped by default (`drop_threshold_cols`).

---

### Chromatography (GC)

`GCData` is the vendor-independent container for gas chromatography: the chromatograms of a series of injections together with their analysis (integrated peaks and amounts per species).

| what | dims | notes |
|---|---|---|
| one chromatogram variable per detector (`fid`, `tcd`) | `(injection, retention_time_<detector>)` | detector units (`pA`, `25 µV`), retention time in s |
| `peak_*` variables (the peak table) | `(peak,)` | injection, detector, species, retention / start / end time in s, height, area |
| `amount`, `area`, `height` | `(injection, species)` | one entry per species and the detector it was quantified on |

The `injection` dimension is always there, with length 1 for a single measurement, and `tos` runs along it. Every detector has its own retention time axis. Injections that differ in run length are padded with `NaN`; injections whose sampling interval differs are never resampled: `group_injections` splits them and `from_ezchrom_rslt_groups` returns one `GCData` per group. The analysis is whatever produced it, named in `attrs["analysis_source"]`.

- **Constructors**: `from_ezchrom_rslt` (one or more Agilent OpenLab EZChrom `.rslt` folders into one series in time order), `from_ezchrom_rslt_groups`, `from_injections` (the common records every vendor converter produces; `to_injections` gives them back), `from_arrays`, `from_netcdf`
- **Accessors**: `detectors`, `n_injections`, `injection`, `sample_names`, `get_unit`, `get_retention_time`, `tos`, `tos_start`, `timestamps`, `get_chromatogram(detector, injection=None)`, `get_peaks()`, `get_amounts(detector=None, quantity="amount")`, `get_evolution(species, detector=None)`
- **Processing**: `select_by_idx`, `select_tos_range`, `select_timestamp_range`, `select_retention_time_range`, the four time-origin methods, `merge` (along the injection axis), `rename_species`
- **Export**: `to_netcdf` (raw counts are packed as int32 with their scale factor, lossless and half the size), `to_dataframe(detector)`

```python
gc_raw = pp.GCData.from_ezchrom_rslt(sorted(gc_dir.glob("0*/*.rslt")))
gc = gc_raw.rename_species()                     # Methanol -> MeOH, ...
tcd = gc.get_chromatogram("tcd")                 # (injection, retention time)
co = gc.get_evolution("CO", detector="tcd")      # amount vs injection, with tos
```

**Agilent OpenLab CDS EZChrom** (`phd_parser.chromatography.agilent.ezchrom`) reads the vendor files directly, without OpenLab or any Agilent library: `.dat` (an OLE2 / Compound File Binary container; `read_dat` returns both detector traces as raw ADC counts with their scale factor and unit, plus the instrument method XML, setup text, audit trail and summary information), `.acaml` / `.acam_` (ACAML 1.5 XML; `read_acaml` returns every peak, the species amounts and the calibration curves), `.macaml` (`read_method_acaml`), `.rst` and `.cfg` (`read_container_strings`). `read_result_set` reads a whole `.rslt` folder and `to_injections` converts it into the common records (SI units, detectors named from the method: `Front Signal (FID)` becomes `fid`). Verified against the AIA netCDF export of the same 120 traces (largest relative deviation 5.8e-8).

---

### XRD

`XRDData` wraps an `xarray.DataArray` with the 2θ angle in degrees throughout. Layouts: `(angle,)` for a single diffractogram and `(scan, angle)` for a series, with optional `tos` and `timestamp` coordinates that every processing step keeps.

- **Constructors**: `from_bruker_xy` (Bruker D8 Advance-ECO, the DIFFRAC.EVA `.xy` export with its header line), `from_e1290` (the same two columns without the header; kept for old notebooks, the name is historical)
- **Accessors**: `angle`, cached `angle_rad`, `theta_deg`, `theta_rad`; `d_spacing(wavelength_angstrom)` and `q_vector(wavelength_angstrom)` via Bragg's law (Cu Kα₁ is 1.5406 Å); `values`, `ndim`, `shape`, `timestamps`, `metadata`
- **Processing**: `sort`, `select_angle_range`, `smooth_savgol`, `smooth_gaussian` (sigma in degrees), `smooth_moving`
- **Analysis**: `fit_reflection(two_theta_deg, window_deg, lineshape)` (one pseudo-Voigt, Gaussian or Lorentzian line from `phd_parser.kernels` plus a linear background; returns a `ReflectionFit`), `scherrer_size(two_theta_deg, fwhm_deg, instrumental_fwhm_deg=...)` (crystallite size in nm; the instrumental width is subtracted in quadrature, the wavelength comes from the file's anode and falls back to 1.5406 Å)

```python
xrd = pp.XRDData.from_bruker_xy(xy_file)                 # anode, operator, time per step in xrd.metadata
fit = xrd.fit_reflection(25.3, window_deg=1.5)           # anatase (101)
size_nm = xrd.scherrer_size(fit.position_deg, fit.fwhm_deg, instrumental_fwhm_deg=0.08)
```

**Reference cards** (`phd_parser.xrd.pdf_card`): ICDD PDF cards exported as tab-separated text are read by `ReferenceCard.from_file` (or `read_pdf_card` for the plain dict). The card number and phase come from the file name (`PDF 78-2486_TiO2-Anatase.txt`). `two_theta(wavelength_angstrom)` recomputes the line positions from the d-spacings for another wavelength, `strongest(n)` picks the strongest lines and `select_two_theta_range` cuts a window, which is what you need to mark reference lines in a plot.

**Bruker DIFFRAC.EVA** (`phd_parser.xrd.bruker.read_xy`): one header line of `Key: "value"` pairs and two columns 2θ / counts. The anode gives the Kα₁ wavelength (`ANODE_WAVELENGTH_ANGSTROM`), which is stored in the attributes next to `source_file` and `anode`.

---

### TGA

`TGAData` holds parallel `temperature` (K) and `mass` (mg) arrays with an optional `mass_init` and an optional `baseline` kept for provenance after blank subtraction.

- **Constructor**: `from_e2290` (Mettler Toledo TGA E2290 `.txt`; `baseline_path` subtracts a blank on load)
- **Accessors**: `mass_fraction` (raises without `mass_init`), `derivative` (dm/dT), `derivative_fraction`
- **Processing**: `cut_front`, `cut_back`, `correct` (subtract a blank interpolated onto the sample's temperature axis), `smooth` (Savitzky-Golay)

The parser `phd_parser.tga.e2290.read_export` splits the file into its five tagged sections (`Curve Name`, `Curve Values`, `Results`, `Sample`, `Method`) and returns the data with the curve name, timestamps, sample name, initial mass and method.

---

### TPR (temperature-programmed reduction)

`TPRData` wraps an `xarray.Dataset` over `time` in s, with `temperature` in °C
and the detector signals (`tcd_h2`, `tcd_o2`, and `fid` where the rig had one)
in mV. The signals keep the sign of the file: on the in-house rig a reduction
is a *positive* excursion of `tcd_h2`, and `scale(signal, -1)` is the explicit
flip when you need the other sign.

- **Constructors**: `from_labview_tcd` (the export of the LabVIEW TPR rig; both column layouts, with an FID column until 2026-07-13 and with `date` / `time` columns from 2026-07-23), `from_arrays`, `from_netcdf`
- **Accessors**: `time`, `temperature`, `signals`, `n_points`, `sample_name`, `sample_weight_mg`, `start_time`, `comment`, `timestamps`, `get_signal`, `integrate(signal, min_s, max_s)`, `heating_rate()`
- **Processing (immutable, recorded)**: `select_time_range`, `select_temperature_range`, `smooth_moving`, `scale`, `per_gram` (divides by the sample weight), `subtract_baseline(signal, "linear" | "constant", time_window_s=... or anchor_points_s=...)`; every subtracted baseline accumulates in `<signal>_baseline`
- **Export**: `to_dataframe`, `to_netcdf`

```python
tpr_raw = pp.TPRData.from_labview_tcd(txt_file)
tpr = (
    tpr_raw
    .subtract_baseline("tcd_h2", "linear", anchor_points_s=[600, 4200])   # before and after the peak
    .per_gram("tcd_h2")
)
consumption = tpr.integrate("tcd_h2")      # mV s g^-1
```

The TCD is not calibrated for H₂, so a consumption is an area in mV s g⁻¹ and
compares runs on the same rig, not absolute amounts. The parser
(`phd_parser.tpr.labview_tcd`, `COLUMN_MAP`) tolerates a decimal comma in the
sample weight and keeps unknown or unnamed columns as further signals with a
warning.

---

### Physisorption (N₂ physisorption / BET)

`PhysisorptionData` is the container for a single isotherm *branch*: relative pressure versus quantity adsorbed (cm³ g⁻¹ STP), wrapped like a single spectrum. A reading that produces both branches returns two instances. The BET fit is modelled directly alongside the curve it was derived from; everything else an instrument reports (t-Plot, BJH, sample log) is preserved verbatim under `.report`.

- **Constructors**: `from_tristar_xls` (Micromeritics TriStar II 3020 multi-report `.XLS`; returns a dict with `"adsorption"` / `"desorption"`), `from_arrays`, `from_netcdf`
- **Accessors**: `relative_pressure`, `values`, `n_points`, `bet`, `surface_area_bet` (m² g⁻¹), `report`, `header`, `sample_name`, `sample_mass_g`; `get_quantity_adsorbed(target_relative_pressure, method, tolerance)`
- **Export**: `to_netcdf`

```python
branches = pp.PhysisorptionData.from_tristar_xls(xls_file)
adsorption = branches["adsorption"]
adsorption.surface_area_bet                     # BET surface area in m^2 g^-1
adsorption.report["analyses"]["t_plot"]         # everything else MicroActive reported
```

**TriStar II 3020** (`phd_parser.physisorption.tristar.read_export`): parses a MicroActive "print selected reports to Excel" export, where every report is its own block of columns separated by a `"|"` divider, with a generic state machine for `key: value` rows, notes and one data table per block. The report labels change capitalisation between exports ("BET Surface Area" in 2026-07, "BET surface area" earlier), so every lookup goes through `tristar.scalar`, which ignores case.

---

## Development

- The *Conventions* section above is the summary for users. The full contract, with the reasons behind each rule (docstrings, naming, the `with_` / `set_` / `del_` trio, time origins, NetCDF, history), is [`CLAUDE.md`](CLAUDE.md), and it binds humans too. Keep the two in sync.
- Tests run against real exports under `tests/`; a new reader comes with a small real file and a test that reads it.
- `uv build` produces the wheel and sdist; the version is in `pyproject.toml` and is recorded in every history entry.

## References

[^1]: Travert, A., & Fernandez, C. (2025). *SpectroChemPy* (Version 0.8.4) [Computer software]. Laboratoire Catalyse and Spectrochemistry (LCS), Normandie Université/CNRS. https://github.com/spectrochempy/spectrochempy (CeCILL-B licence)
