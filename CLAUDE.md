# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Knowledge vault (Obsidian): read this before asking

Julius keeps his PhD knowledge notes in an Obsidian vault at

`C:\Users\jsommer1\Documents\MyPhD\`

It sits outside every code repo and holds the context the code does not record:
background, decisions, literature notes, project and programme structure. When a
question is about *why* rather than *how the code works*, look there first. The
link is two-way: the vault's own header block lists `C:\Users\jsommer1\CODE` and
all four packages, so it expects you to come back here for the code.

**Entry points, in order.** The vault root has its own `CLAUDE.md` (it has to
live at root level), which does nothing but forward to `me.md`. `me.md` is the
briefing: who Julius is, how he wants to be worked with, and how the vault is
laid out. `me.md`, `Vault Map` and `Skills Map` all open with the same header
block, so any of the three orients you.

| file | what it is |
|---|---|
| `CLAUDE.md` | one line, points at `me.md` |
| `me.md` | the briefing and his working preferences (start here) |
| `000 AIOS/Maps/Vault Map.md` | folder structure and the AI tag system |
| `000 AIOS/Maps/Skills Map.md` | index of the skills the AI can run on his behalf |
| `Atlas.md` | topic map of the subject-matter notes |
| `000 AIOS/History/` | where AI-written notes go |
| `000 AIOS/Skills/<name>/` | one self-contained folder per skill: the note is the contract, the assets beside it are the implementation |

**Read those core notes from disk, via the shell, every time.** `me.md` asks for
this explicitly: never work from a cached or staged copy, and re-read
immediately before writing, because a stale copy has already nearly clobbered
his edits.

**Vault layout** (LYT ACE, plus AIOS): `+/` inbox for new captures, `+ Vault/`
imported sources (`+ Vault/Zotero/` named `@{first-author}.{three-word-title}.{year}`),
`100 Notes/` timeless notes (maps, people, quotes), `200 Calendar/` days and
reviews, `300 Efforts/` areas, projects and works, `x/` templates, attachments,
PDFs, images, scripts, `000 AIOS/` the AI operating system itself.

**Writing rules, from `me.md` and the Vault Map.** Read anything. Only *create*
notes inside `000 AIOS/History/`, tagged `ai/monitored`. Keep AI-written and
hand-written notes clearly separated. Whenever a note is touched beyond reading,
set or correct its AI tag:

| tag | when |
|---|---|
| `ai/checked` | language or consistency pass, no deeper content change |
| `ai/assisted` | more than cosmetic: parts written, formulas changed or adjusted |
| `ai/monitored` | AI maintains part or all of the note on a regular basis |

**Link generously, but only to notes that exist.** Wikipedia rule: link the
first meaningful mention, not every occurrence. If a link would point at a note
that does not exist yet, flag it and let Julius decide rather than creating it
silently.

`me.md` also states general working preferences, and those are not vault-only.
They apply to work in this repo as well, so read it rather than guessing.

## Commands

```bash
# Install in editable mode (run once)
uv pip install -e .

# Run all tests
pytest

# Run a single test file
pytest tests/spectroscopy/raman/test_raman_data.py

# Run a single test by name
pytest tests/spectroscopy/raman/test_raman_data.py::test_function_name
```

The project uses `uv` for dependency management. The lockfile is `uv.lock`; add new dependencies via `uv add <package>`.

## Docstrings

All public classes, methods, functions, and properties must have a NumPy/SciPy-style docstring:

```python
def example(param1: int, param2: str = "default") -> list:
    """Short one-line summary.

    Optional extended description (only when non-obvious).

    Parameters
    ----------
    param1 : int
        Description of param1.
    param2 : str, optional
        Description of param2 (default is "default").

    Returns
    -------
    list
        Description of the return value.

    Raises
    ------
    ValueError
        When and why this is raised.
    """
```

Rules:
- One-line summary on the opening `"""` line.
- Skip the `Parameters` section for zero-argument methods/properties.
- Skip the `Returns` section for `None`-returning methods.
- `Raises` only when the function actually raises.
- Private helpers (leading `_`) do not need docstrings.

## Naming

Use full, unabbreviated names for fields, columns, properties, and variables — `relative_pressure`, not `p_rel`; `quantity_adsorbed`, not `q`; `temperature`, not `temp`. Established names of the technique or method itself are fine to keep as-is (`BET`, `BJH`, `XRD`, `IR`, `MS`, `TGA`) — these are not abbreviations of a longer phrase, they're what the method is actually called.

### Species, not components

A chemical species is always called **species**: in variables, dims, coords,
columns, record keys and docstrings (`species`, `species_name`,
`species_detector`, `peak_species`, `rename_species`). Never **component**,
which in this lab also means a mechanical part of a setup (and stays correct
for mathematical components, e.g. quadrature components). Not **compound**
either: that is the vendor's word, kept only where a vendor format names it
(the ACAML `Compound` tags, the Compound File Binary container).

Species names follow a naming convention, see *Species naming conventions*
below. Parsers keep the names exactly as the instrument wrote them; renaming
is an explicit `rename_species()` step, never done silently on read.

### Combining two datasets

Three different operations, three different names — never mix them up:

- **extend** — joining along the energy-related axis (wavenumber, Raman shift, 2θ, m/z), i.e. widening the spectral range.
- **merge** — joining along the *other* axis (scan/time), i.e. appending more scans of the same spectral range. `IRData.merge` / `IRData.merge_all`.
- **vstack** — taking already 2-D data and stacking it along a *new*, additional dimension.

## Text & plot style conventions

Applies to code comments, docstrings, printed/repr output, and figures (titles, axis labels, tick labels, legends, annotations). The same rule set lives in `me.md` in the Obsidian vault and in the sibling repos' `CLAUDE.md` files (`phd_execute`, `phd_ui`, `phd_models`); keep them in sync if this changes.

- No em dashes ("—"). Use a comma, period, or separate sentence instead.
- Capitalize only in headers/titles, proper nouns, or where otherwise grammatically correct. Axis labels, tick labels, legend entries, and other in-plot text are lowercase by default.
- Axis labels use the format `quantity / unit` (the quantity divided by its unit), e.g. `time / s`, `temperature / K`. If the quantity is dimensionless, use an en dash "–" in place of the unit, e.g. `fraction / –`.
- Units and exponents use power notation, not slashes: `m^2 s^-1`, `mol m^-3 s^-1`, not `m^2/s`, `mol/m^3/s`. Applies in code, plots, printed output, and LaTeX math. In Obsidian notes, `\frac{}{}` fraction style is fine in math environments; code and plots still use the power form.
- Chemical species, formulas and reactions use the mhchem `\ce{}` package, always inside math: `$\ce{CO2}$`, `$\ce{Cu^0}$`, `$\ce{Al^3+}$`, `$\ce{CO2 + 3H2 <=> CH3OH + H2O}$`. Applies to Obsidian notes and LaTeX (thesis). Plain text stays only where math cannot render: file names/note titles, wikilinks, frontmatter, and code/plot labels (matplotlib has no mhchem).

## Architecture

**PhD_parser** is a data-parsing library for experimental equipment used in catalysis research (Urakawa group, TU Delft). It is under active development and the API changes frequently.

### Module layout

Each instrument type lives in its own subpackage under `src/phd_parser/`:

| Subpackage | Core class | Backing store |
|---|---|---|
| `labview` | `LVData` | `xr.Dataset` |
| `xrd` | `XRDData` | `xr.DataArray` |
| `spectroscopy` | `SpectroscopyData`; thin `IRData` (`infrared/`) and `RamanData` (`raman/`); `xps/` reads files only | `xr.Dataset` |
| `tga` | `TGAData` | two NumPy arrays |
| `massspec` | `MSData` | `xr.Dataset` |
| `physisorption` | `PhysisorptionData` | `xr.Dataset` |
| `chromatography` | `GCData` | `xr.Dataset` |
| `naming` | *(no data class)*; `species` conventions and `rename_species` | JSON files |

Every subpackage follows the same structure:
- `core.py` — the Pydantic `BaseModel` wrapper class
- one parser module per instrument/software (e.g. `renishaw.py`, `omnic.py`, `quadstar.py`)

### Core design patterns

**Pydantic `BaseModel` wrappers.** All core classes subclass `pydantic.BaseModel`. The xarray object is stored as a field; field validators handle construction-time checks.

**Core classes are re-exported at the package root.** `pp.GCData is pp.chromatography.GCData`, and the same for every core class plus `rename_species` / `SpeciesConvention`. A new core class gets its line in `phd_parser/__init__.py` (and in `tests/test_top_level_imports.py`) in the same change.

**Immutable processing API.** Every processing method (smoothing, baseline correction, normalization, selection) returns a *new* instance rather than mutating `self`. Use chaining: `ir.normalise_max().smooth()`.

**Classmethod constructors, not `__init__`.** Data is always loaded via a `from_*` classmethod (e.g. `RamanData.from_renishaw_wdf`, `MSData.from_quadstar_asc`). Each classmethod calls the corresponding parser, then passes the parsed dict into the model.

**Parser convention.** Parser modules expose one or more functions that return a plain dict with at minimum a `"data"` key (an xarray object or dict of arrays) and a `"meta"` key (a dict of provenance metadata). Parsers do not return model instances.

**Cached properties for derived quantities.** Expensive conversions (Raman shift per cm⁻¹, XRD d-spacing, IR wavenumber per cm⁻¹, MS trace indexing) are implemented as `@cached_property` so they are computed once and reused.

**`with_` / `set_` / `del_` mean specific things.** `with_X` applies X *and* recalculates whatever else must change so the physical meaning of the data is preserved (`with_background` recomputes the values; `with_tos_start` rebases `tos` so the absolute timestamps survive). `set_X` force-assigns X and recalculates nothing. `del_X` removes X and leaves the data alone. Keep this trio consistent when adding new state to a class.

**One time origin, four operations.** Every class with a `tos_start` (`LVData`, `IRData`, `MSData`) exposes `with_tos_start` / `set_tos_start` / `del_tos_start` / `move_tos_start_by(delta)`, with identical semantics and docstrings. `move_tos_start_by` is defined as `with_tos_start(tos_start + delta)`, so a *later* origin means *smaller* `tos` values. `RamanData` has them through `SpectroscopyData`; `XRDData` stores absolute timestamps as a coordinate instead.

**Two ways to cut a time window.** The same three classes expose `select_tos_range(min_s, max_s)` (elapsed seconds) and `select_timestamp_range(start, end)` (absolute wall clock — `pd.Timestamp`, `datetime`, `np.datetime64` or a parseable string). The absolute form converts its bounds against `tos_start` and delegates to the elapsed form, so the two never drift apart; it raises if no `tos_start` is set. A timezone-naive bound is read as wall-clock time *in the origin's timezone* — never as UTC — because the timestamps in the log book are local. Keep this pair consistent when adding another time-anchored class.

**Time comes from the instrument, not from the filename.** OMNIC's `... at 0,90 Hours.spa` names are rounded to 0.01 h (36 s) and restart at zero for every new series, so a filename-derived `tos` silently stacks two measurements on top of each other. Both `.spa` readers take the per-scan acquisition timestamp recorded inside the file (`omnic.read_spa_datetime`); filenames are a last-resort fallback that warns. The same rule applies to any parser: prefer a recorded absolute time over anything reconstructed from a name or an index.

**Instrument exports drift over time.** The same setup gains or renames a column between runs (e.g. LabView b67 box 5 gained `F1 CO PV` in 2026-07). A parser must read old *and* new files, and a directory may hold both: concatenate on the union of the columns and leave NaN where a file did not record a channel. An unrecognised column is skipped with a warning naming it — never a hard failure, and never silently carried through as un-parsed strings.

### Spectral axis conventions

Each module stores its primary axis in SI units; convenience properties expose common alternative units:

- **Raman**: the shift is the `wavenumber` coordinate in m⁻¹ (Stokes positive), as for IR; `.shift` / `.shift_per_cm` are aliases, `.scattered_wavenumber` the absolute scattered light
- **XRD**: `angle` coordinate in degrees (2θ); `.d_spacing` and `.q_vector` depend on stored wavelength (default Cu Kα₁ 1.5406 Å)
- **IR**: `wavenumber` coordinate in m⁻¹; `.wavenumber_per_cm` for cm⁻¹
- **TGA**: temperature in K, mass in mg; `.mass_fraction` normalises to `mass_init`

### Processing history

Every public method of a core class that returns a new object, and every
`from_*` constructor, is decorated with `phd_parser.history.records_step`. It
appends `{"step", "parameters", "time", "version"}` to `ds.attrs["history"]`
of the result, never of the input, and only for the outermost call (a step
calling other recorded steps shows up once). A new processing method gets the
decorator in the same change; a getter does not. A caller that builds its
result from recorded steps but should be recorded as one (e.g.
`GCData.from_ezchrom_rslt_groups`) uses `suppress_recording()` and
`append_history()`. `LVData` keeps the history in `metadata` too, via
`_with_history`. The history is plain JSON data; `to_netcdf` must write it
(`encode_attrs` / `decode_attrs`, prefix `json:`), and a new class with
`to_netcdf` gets a round-trip test of it in `tests/test_history.py`.

### Spectroscopy layout

IR and Raman are one data model. `spectroscopy/core.py` holds
`SpectroscopyData`, which owns every processing step, the background and
`data_type` conversions included; the axis is always the `wavenumber` dim in
m⁻¹ (the Raman shift for Raman). `IRData` (`spectroscopy/infrared/`) and
`RamanData` (`spectroscopy/raman/`) are thin subclasses: they set the class
variables `_technique` (written to `ds.attrs["technique"]` by `_build_ds`) and
`_default_data_type`, and add readers, community names (`shift`,
`select_shift_range`, `get_evolution(shift_per_cm=...)`) and Raman's
excitation wavelength and RamanSPy conversion. A new processing step goes into
`SpectroscopyData`, never into a subclass, uses `self._new(ds=...)` to keep the
caller's class and fields, and `self._carry_auxiliary` /
`self._slice_auxiliary_to` to carry the background and map coordinates.
`merge` rebases only `BACKGROUND_REFERENCED_TYPES`. `spectroscopy/xps/` is a
reader only. The old import paths `phd_parser.infrared|raman|xps[.module]`
are aliased to the same module objects in `phd_parser/__init__.py`; write new
code against `phd_parser.spectroscopy`.

### MS block structure

`MSData` wraps an `xr.Dataset` with a shared `cycle` dimension. `block_0` holds m/z channels (`cycle × mz`); additional datablocks hold auxiliary sensors (`cycle × ch_N`). Trace corrections are also logged in `ds.attrs["trace_corrections"]`, next to the general history.

### IR background and merging

`IRData` keeps the background as a separate 1-D `single_beam` variable (`ds["background"]`), never folded into the data, so `data_type` conversions and background switches are reversible. `merge` (scan axis) leans on exactly that: two measurements are only comparable across a spectrometer restart in raw detector units, so background-dependent segments are converted with `to_single_beam()`, merged, and converted back against the surviving background (`convert_to_single_beam=False` refuses instead; segments already sharing a background skip the round trip). It keeps exactly one background (the pre-experiment one, i.e. the chronologically first segment's), orders segments by absolute timestamp rather than by `tos`, and rebases the later segment's `tos` onto the first's `tos_start`.

### Chromatography layout

`GCData` (`chromatography/core.py`) is the vendor-independent core class, one
subpackage per vendor holds the parsers (`chromatography/agilent/ezchrom.py`).
The contract between them is the list of common *injection records*
(`GCData.from_injections` documents the keys); each vendor module has a
`to_injections()` that produces them, so a new vendor never touches the core.

- The `injection` dim is always present (length 1 for a single measurement); `tos` runs along it.
- Every detector has its own `retention_time_<detector>` dim, even when the axes are identical, so one code path covers instruments that sample detectors differently.
- Different run lengths are padded with NaN. A different sampling interval within one detector is never resampled: `group_injections` splits such records and `from_injections` raises, pointing there.
- The analysis (peak table along `peak`, amounts along `(injection, species)`) is one per object and named in `attrs["analysis_source"]`. An own integration returns a new object; the vendor's stays on the original.
- `to_netcdf` writes NetCDF3 through scipy when neither `netCDF4` nor `h5netcdf` is installed (the case in this venv), which cannot store non-ASCII attribute text or string lists: those are JSON-encoded, int64 comes back as int32 and is cast back on load.

### Species naming conventions

`phd_parser.naming.species` maps the spellings of a species onto one canonical
name. A convention is human-readable JSON; the standard one,
`naming/species/standard.json`, uses the canonical names of `phd_models`
(`CO2`, `MeOH`, `EtOH`, `PrOH`, ...). Matching ignores case, spaces, hyphens,
underscores and dots and reads subscript digits as digits; a convention where
two species claim one spelling is refused on load. Every function taking a
convention accepts `None` (the standard), a JSON path, an already read dict
(full form, or the flat `{canonical: [aliases]}` form of `phd_ui`), or a
`SpeciesConvention`. Add a spelling to `standard.json` rather than mapping it
in a notebook.

### LabView channel metadata

`LVData` stores per-channel metadata (unit, group, species, location) in `xr.DataArray.attrs`. The single dimension is `tos` (elapsed seconds since run start). Channels absent from `b67box5.CHANNEL_META` are dropped on read (see *Instrument exports drift over time*); adding a channel means adding its metadata entry.

### Physisorption isotherm shape

`PhysisorptionData` wraps a *single* isotherm branch — one `xr.DataArray` named `quantity_adsorbed`, dims `("relative_pressure",)` — exactly like a single Raman spectrum or XRD pattern elsewhere in this repo. There is no "branch" concept baked into the type: a reading that produces both adsorption and desorption (e.g. `from_tristar_xls`) returns a `dict[str, PhysisorptionData]` with keys `"adsorption"`/`"desorption"` instead of one dual-branch object. The BET fit (conventionally derived from the adsorption branch) is modelled directly as `.attrs["bet"]`, since every physisorption instrument and analysis method agrees on it. Everything else a parser extracts (t-Plot, BJH, sample log, instrument header) is preserved verbatim under `.attrs["report"]` (JSON-encoded so both survive NetCDF round-trips) rather than given dedicated properties. Parser modules (e.g. `tristar.py`) should still extract as much as the source format offers; the core class only grows new typed accessors when one is actually needed.
