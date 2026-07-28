# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode (run once)
uv pip install -e .

# Run all tests
pytest

# Run a single test file
pytest tests/raman/test_core.py

# Run a single test by name
pytest tests/raman/test_core.py::test_function_name
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

### Combining two datasets

Three different operations, three different names — never mix them up:

- **extend** — joining along the energy-related axis (wavenumber, Raman shift, 2θ, m/z), i.e. widening the spectral range.
- **merge** — joining along the *other* axis (scan/time), i.e. appending more scans of the same spectral range. `IRData.merge` / `IRData.merge_all`.
- **vstack** — taking already 2-D data and stacking it along a *new*, additional dimension.

## Architecture

**PhD_parser** is a data-parsing library for experimental equipment used in catalysis research (Urakawa group, TU Delft). It is under active development and the API changes frequently.

### Module layout

Each instrument type lives in its own subpackage under `src/phd_parser/`:

| Subpackage | Core class | Backing store |
|---|---|---|
| `labview` | `LVData` | `xr.Dataset` |
| `raman` | `RamanData` | `xr.DataArray` |
| `xrd` | `XRDData` | `xr.DataArray` |
| `infrared` | `IRData` | `xr.Dataset` |
| `tga` | `TGAData` | two NumPy arrays |
| `massspec` | `MSData` | `xr.Dataset` |
| `physisorption` | `PhysisorptionData` | `xr.Dataset` |
| `xps` | *(no core class yet)* | — |

Every subpackage follows the same structure:
- `core.py` — the Pydantic `BaseModel` wrapper class
- one parser module per instrument/software (e.g. `renishaw.py`, `omnic.py`, `quadstar.py`)

### Core design patterns

**Pydantic `BaseModel` wrappers.** All core classes subclass `pydantic.BaseModel`. The xarray object is stored as a field; field validators handle construction-time checks.

**Immutable processing API.** Every processing method (smoothing, baseline correction, normalization, selection) returns a *new* instance rather than mutating `self`. Use chaining: `ir.normalise_max().smooth()`.

**Classmethod constructors, not `__init__`.** Data is always loaded via a `from_*` classmethod (e.g. `RamanData.from_renishaw_wdf`, `MSData.from_quadstar_asc`). Each classmethod calls the corresponding parser, then passes the parsed dict into the model.

**Parser convention.** Parser modules expose one or more functions that return a plain dict with at minimum a `"data"` key (an xarray object or dict of arrays) and a `"meta"` key (a dict of provenance metadata). Parsers do not return model instances.

**Cached properties for derived quantities.** Expensive conversions (Raman shift per cm⁻¹, XRD d-spacing, IR wavenumber per cm⁻¹, MS trace indexing) are implemented as `@cached_property` so they are computed once and reused.

**`with_` / `set_` / `del_` mean specific things.** `with_X` applies X *and* recalculates whatever else must change so the physical meaning of the data is preserved (`with_background` recomputes the values; `with_tos_start` rebases `tos` so the absolute timestamps survive). `set_X` force-assigns X and recalculates nothing. `del_X` removes X and leaves the data alone. Keep this trio consistent when adding new state to a class.

**One time origin, four operations.** Every class with a `tos_start` (`LVData`, `IRData`, `MSData`) exposes `with_tos_start` / `set_tos_start` / `del_tos_start` / `move_tos_start_by(delta)`, with identical semantics and docstrings. `move_tos_start_by` is defined as `with_tos_start(tos_start + delta)`, so a *later* origin means *smaller* `tos` values. `RamanData` has `tos` but no origin; `XRDData` stores absolute timestamps as a coordinate instead.

**Time comes from the instrument, not from the filename.** OMNIC's `... at 0,90 Hours.spa` names are rounded to 0.01 h (36 s) and restart at zero for every new series, so a filename-derived `tos` silently stacks two measurements on top of each other. Both `.spa` readers take the per-scan acquisition timestamp recorded inside the file (`omnic.read_spa_datetime`); filenames are a last-resort fallback that warns. The same rule applies to any parser: prefer a recorded absolute time over anything reconstructed from a name or an index.

**Instrument exports drift over time.** The same setup gains or renames a column between runs (e.g. LabView b67 box 5 gained `F1 CO PV` in 2026-07). A parser must read old *and* new files, and a directory may hold both: concatenate on the union of the columns and leave NaN where a file did not record a channel. An unrecognised column is skipped with a warning naming it — never a hard failure, and never silently carried through as un-parsed strings.

### Spectral axis conventions

Each module stores its primary axis in SI units; convenience properties expose common alternative units:

- **Raman**: `shift` coordinate in m⁻¹ (Stokes positive); `.shift_per_cm` for cm⁻¹
- **XRD**: `angle` coordinate in degrees (2θ); `.d_spacing` and `.q_vector` depend on stored wavelength (default Cu Kα₁ 1.5406 Å)
- **IR**: `wavenumber` coordinate in m⁻¹; `.wavenumber_per_cm` for cm⁻¹
- **TGA**: temperature in K, mass in mg; `.mass_fraction` normalises to `mass_init`

### MS block structure

`MSData` wraps an `xr.Dataset` with a shared `cycle` dimension. `block_0` holds m/z channels (`cycle × mz`); additional datablocks hold auxiliary sensors (`cycle × ch_N`). Processing history is appended to `ds.attrs["trace_corrections"]`.

### IR background and merging

`IRData` keeps the background as a separate 1-D `single_beam` variable (`ds["background"]`), never folded into the data, so `data_type` conversions and background switches are reversible. `merge` (scan axis) leans on exactly that: two measurements are only comparable across a spectrometer restart in raw detector units, so background-dependent segments are converted with `to_single_beam()`, merged, and converted back against the surviving background (`convert_to_single_beam=False` refuses instead; segments already sharing a background skip the round trip). It keeps exactly one background (the pre-experiment one, i.e. the chronologically first segment's), orders segments by absolute timestamp rather than by `tos`, and rebases the later segment's `tos` onto the first's `tos_start`.

### LabView channel metadata

`LVData` stores per-channel metadata (unit, group, species, location) in `xr.DataArray.attrs`. The single dimension is `tos` (elapsed seconds since run start). Channels absent from `b67box5.CHANNEL_META` are dropped on read (see *Instrument exports drift over time*); adding a channel means adding its metadata entry.

### Physisorption isotherm shape

`PhysisorptionData` wraps a *single* isotherm branch — one `xr.DataArray` named `quantity_adsorbed`, dims `("relative_pressure",)` — exactly like a single Raman spectrum or XRD pattern elsewhere in this repo. There is no "branch" concept baked into the type: a reading that produces both adsorption and desorption (e.g. `from_tristar_xls`) returns a `dict[str, PhysisorptionData]` with keys `"adsorption"`/`"desorption"` instead of one dual-branch object. The BET fit (conventionally derived from the adsorption branch) is modelled directly as `.attrs["bet"]`, since every physisorption instrument and analysis method agrees on it. Everything else a parser extracts (t-Plot, BJH, sample log, instrument header) is preserved verbatim under `.attrs["report"]` (JSON-encoded so both survive NetCDF round-trips) rather than given dedicated properties. Parser modules (e.g. `tristar.py`) should still extract as much as the source format offers; the core class only grows new typed accessors when one is actually needed.
