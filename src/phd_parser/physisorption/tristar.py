"""Parser for Micromeritics TriStar II 3020 multi-report ``.XLS`` exports.

MicroActive's "print selected reports to Excel" produces a single sheet
where every selected report (Isotherm, BET, t-Plot, BJH, ...) is laid out
as its own block of columns, separated by a literal ``"|"`` divider column,
all sharing the same row grid. Each block internally follows the same
micro-pattern: a title row, ``key: value`` metadata rows, free-text notes,
and at most one data table (header row + rows until the next blank row).

This module extracts every block generically and returns the full report
as a nested dict (``meta``); :mod:`phd_parser.physisorption.core` only
consumes the isotherm out of it to build the minimal, instrument-agnostic
``PhysisorptionData`` model.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ACCEPTED_FILE_EXTENSIONS = {".xls"}

SECTION_TITLES = [
    "Summary Report",
    "Isotherm Tabular Report",
    "Isotherm Linear Plot",
    "Isotherm Log Plot",
    "BET Report",
    "BET Surface Area Plot",
    "t-Plot Report",
    "t-Plot",
    "BJH Adsorption Pore Distribution Report",
    "BJH Adsorption Cumulative Pore Volume (Smaller)",
    "BJH Adsorption dV/dlog(w) Pore Volume",
    "BJH Desorption Pore Distribution Report",
    "BJH Desorption Cumulative Pore Volume (Smaller)",
    "BJH Desorption dV/dlog(w) Pore Volume",
    "Sample log",
]

_DATE_RE = re.compile(r"^\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
_RANGE_RE = re.compile(
    r"^([+-]?[\d.]+(?:[eE][+-]?\d+)?)\s*(\S*?)\s+to\s+([+-]?[\d.]+(?:[eE][+-]?\d+)?)\s*(\S*)$"
)
_ERR_RE = re.compile(
    r"^([+-]?[\d.]+(?:[eE][+-]?\d+)?)\s*\xb1\s*([+-]?[\d.]+(?:[eE][+-]?\d+)?)\s*(.*)$"
)
_NUM_RE = re.compile(r"^([+-]?[\d.]+(?:[eE][+-]?\d+)?)\s*(.*)$")


def _parse_value(raw: Any) -> Any:
    """Convert a single report cell into a Python value.

    Recognises ``"value ± error unit"``, ``"value unit"``, ``"lo unit to
    hi unit"`` ranges, ``"dd-mm-yy HH:MM:SS"`` timestamps, ``"Yes"``/``"No"``
    booleans, and the ``"*"`` not-reported marker. Anything else is returned
    unchanged (already-numeric cells, or free-text strings such as sample
    names).

    Parameters
    ----------
    raw : Any
        Raw cell value as read by :func:`pandas.read_excel`.

    Returns
    -------
    Any
        A ``float``, ``bool``, :class:`pandas.Timestamp`, ``dict`` (for
        value/error/unit or range cells), the original string, or ``None``.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        return None
    if text in ("Yes", "No"):
        return text == "Yes"
    if text == "*":
        return None
    if _DATE_RE.match(text):
        return pd.to_datetime(text, format="%d-%m-%y %H:%M:%S")
    m = _RANGE_RE.match(text)
    if m:
        lo, lo_unit, hi, unit = m.groups()
        return {"low": float(lo), "high": float(hi), "unit": (unit or lo_unit) or None}
    m = _ERR_RE.match(text)
    if m:
        val, err, unit = m.groups()
        return {"value": float(val), "error": float(err), "unit": unit.strip() or None}
    m = _NUM_RE.match(text)
    if m:
        val, unit = m.groups()
        unit = unit.strip()
        return {"value": float(val), "unit": unit} if unit else float(val)
    return text


def _is_blank_row(row: list) -> bool:
    return all(v is None or (isinstance(v, float) and np.isnan(v)) for v in row)


def _is_kv_row(row: list) -> bool:
    first = row[0]
    return isinstance(first, str) and first.rstrip().endswith(":")


def _is_header_row(row: list) -> bool:
    non_null = [v for v in row if not (v is None or (isinstance(v, float) and np.isnan(v)))]
    return len(non_null) >= 2 and all(isinstance(v, str) for v in non_null)


def _non_null(row: list) -> list:
    return [v for v in row if not (v is None or (isinstance(v, float) and np.isnan(v)))]


def _parse_block_rows(rows: list[list]) -> dict[str, Any]:
    """Run the generic kv/table/note state machine over one block's rows.

    Some sections caption a table with one or more extra header-like rows
    (e.g. a sample-name row) directly above the real column-header row; of
    any run of consecutive header-like rows, only the last is treated as
    the real header — the rest become notes.

    Parameters
    ----------
    rows : list of list
        Raw cell rows for a single report block, starting right after its
        title row.

    Returns
    -------
    dict
        Dict with keys ``"scalars"`` (key → parsed value), ``"table"``
        (``{"columns": [...], "rows": [[...], ...]}`` or ``None`` if no
        table was found), and ``"notes"`` (list of free-text strings).
    """
    scalars: dict[str, Any] = {}
    notes: list[str] = []
    table: Optional[dict[str, Any]] = None
    pending_header: Optional[list[Optional[str]]] = None
    pending_rows: list[list] = []

    is_blank = [_is_blank_row(r) for r in rows]
    is_header = [_is_header_row(r) for r in rows]

    n = len(rows)
    i = 0
    while i < n:
        row = rows[i]
        if is_blank[i]:
            if pending_header is not None and pending_rows:
                table = {"columns": pending_header, "rows": pending_rows}
            pending_header, pending_rows = None, []
            i += 1
            continue
        if pending_header is not None:
            pending_rows.append(row)
            i += 1
            continue
        if _is_kv_row(row):
            key = row[0].rstrip(":").strip()
            scalars[key] = _parse_value(row[1]) if len(row) > 1 else None
            i += 1
            continue
        if is_header[i]:
            next_is_header = i + 1 < n and is_header[i + 1]
            after_next_is_header = i + 2 < n and is_header[i + 2]
            if next_is_header and not after_next_is_header:
                # Exactly one caption row directly above the real header
                # (e.g. a sample-name row) — note it, the next row is the
                # real header. A longer run (e.g. string-typed data rows
                # that happen to look header-like, as in the Sample log)
                # is left alone: this row is taken as the real header.
                notes.append(" | ".join(str(v) for v in _non_null(row)))
            else:
                pending_header = [
                    str(v) if not (v is None or (isinstance(v, float) and np.isnan(v))) else None
                    for v in row
                ]
            i += 1
            continue
        non_null = _non_null(row)
        if len(non_null) == 1 and isinstance(non_null[0], str):
            notes.append(non_null[0])
        i += 1

    if pending_header is not None and pending_rows:
        table = {"columns": pending_header, "rows": pending_rows}

    return {"scalars": scalars, "table": table, "notes": notes}


def _find_divider_columns(df: pd.DataFrame) -> list[int]:
    counts = (df == "|").sum(axis=0)
    return sorted(int(c) for c in counts.index[counts > 0])


def _block_ranges(df: pd.DataFrame, dividers: list[int]) -> list[tuple[int, int]]:
    starts = [0] + [d + 1 for d in dividers]
    ends = dividers + [df.shape[1]]
    return list(zip(starts, ends))


def _find_title_row(df: pd.DataFrame, col: int) -> tuple[Optional[int], Optional[str]]:
    column = df.iloc[:, col]
    for title in SECTION_TITLES:
        matches = column.index[column == title]
        if len(matches):
            return int(matches[0]), title
    return None, None


# Maps each "property/analysis" to its `*Report` section (scalars, and a
# table for analyses whose fit points are reported alongside the results)
# and the further sections that are just additional plot views of the same
# analysis.
ANALYSIS_GROUPS: dict[str, dict[str, Any]] = {
    "isotherm": {
        "report": "Isotherm Tabular Report",
        "plots": {"linear": "Isotherm Linear Plot", "log": "Isotherm Log Plot"},
    },
    "bet": {
        "report": "BET Report",
        "plots": {"surface_area": "BET Surface Area Plot"},
    },
    "t_plot": {
        "report": "t-Plot Report",
        "plots": {"thickness_curve": "t-Plot"},
    },
    "bjh_adsorption": {
        "report": "BJH Adsorption Pore Distribution Report",
        "plots": {
            "cumulative_pore_volume": "BJH Adsorption Cumulative Pore Volume (Smaller)",
            "dvdlogw_pore_volume": "BJH Adsorption dV/dlog(w) Pore Volume",
        },
    },
    "bjh_desorption": {
        "report": "BJH Desorption Pore Distribution Report",
        "plots": {
            "cumulative_pore_volume": "BJH Desorption Cumulative Pore Volume (Smaller)",
            "dvdlogw_pore_volume": "BJH Desorption dV/dlog(w) Pore Volume",
        },
    },
}


def read_export(filepath: Union[str, Path]) -> dict[str, Any]:
    """Parse a Micromeritics TriStar II 3020 multi-report ``.XLS`` export.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.XLS`` file produced by MicroActive's "print selected
        reports to file" feature.

    Returns
    -------
    dict
        Dict with keys:

        ``"data"`` : dict
            Isotherm arrays: ``"p_rel_ads"``, ``"q_ads"``, ``"p_rel_des"``,
            ``"q_des"`` (``numpy.ndarray`` of float, branch omitted if the
            Isotherm Tabular Report section is absent).
        ``"bet"`` : dict or None
            Cleaned BET scalar results (see :func:`_extract_bet`), or
            ``None`` if no BET Report section was found.
        ``"meta"`` : dict
            ``"header"`` (file/sample/instrument metadata shared by every
            block), ``"summary"`` (the Summary Report section, usually
            only populated when multiple samples were compared),
            ``"analyses"`` (dict keyed by analysis name — ``"isotherm"``,
            ``"bet"``, ``"t_plot"``, ``"bjh_adsorption"``,
            ``"bjh_desorption"`` — each a ``{"report", "plots"}`` dict;
            ``"report"`` and each entry of ``"plots"`` is a
            ``{"scalars", "table", "notes"}`` dict as produced by
            :func:`_parse_block_rows`), and ``"sample_log"`` (the Sample
            log section).

    Raises
    ------
    ValueError
        If ``filepath`` does not have a ``.xls`` extension, or no report
        sections could be located in the sheet.
    """
    filepath = Path(filepath)
    if filepath.suffix.lower() not in ACCEPTED_FILE_EXTENSIONS:
        raise ValueError(f"File must be a .xls file, got {filepath.suffix!r}")

    df = pd.read_excel(filepath, sheet_name=0, header=None)
    dividers = _find_divider_columns(df)
    blocks = _block_ranges(df, dividers)

    sections: dict[str, Any] = {}
    title_rows: list[int] = []
    for col_start, col_end in blocks:
        title_row, title = _find_title_row(df, col_start)
        if title is None:
            logger.warning("Could not identify a known section title at column %d", col_start)
            continue
        title_rows.append(title_row)
        block_df = df.iloc[title_row + 1 :, col_start:col_end]
        last_idx = block_df.dropna(how="all").index.max()
        if pd.isna(last_idx):
            sections[title] = {"scalars": {}, "table": None, "notes": []}
            continue
        rows = df.iloc[title_row + 1 : last_idx + 1, col_start:col_end].values.tolist()
        sections[title] = _parse_block_rows(rows)

    if not sections:
        raise ValueError("No known report sections found in the sheet.")

    header_row_end = min(title_rows) if title_rows else df.shape[0]
    header_block_end = blocks[0][1]
    header_rows = df.iloc[:header_row_end, blocks[0][0] : header_block_end].values.tolist()
    header = _parse_block_rows(header_rows)

    analyses: dict[str, Any] = {}
    for name, spec in ANALYSIS_GROUPS.items():
        report_title = spec["report"]
        if report_title not in sections:
            continue
        analyses[name] = {
            "report": sections[report_title],
            "plots": {
                plot_key: sections[title]
                for plot_key, title in spec["plots"].items()
                if title in sections
            },
        }

    data = _extract_isotherm(analyses.get("isotherm", {}).get("report"))
    bet = _extract_bet(analyses.get("bet", {}).get("report"))

    return {
        "data": data,
        "bet": bet,
        "meta": {
            "header": header,
            "summary": sections.get("Summary Report"),
            "analyses": analyses,
            "sample_log": sections.get("Sample log"),
        },
    }


def _scalar_value(scalars: dict[str, Any], key: str) -> Any:
    v = scalars.get(key)
    return v.get("value") if isinstance(v, dict) else v


def _scalar_error(scalars: dict[str, Any], key: str) -> Optional[float]:
    v = scalars.get(key)
    return v.get("error") if isinstance(v, dict) else None


def _scalar_unit(scalars: dict[str, Any], key: str) -> Optional[str]:
    v = scalars.get(key)
    return v.get("unit") if isinstance(v, dict) else None


def _extract_bet(section: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Pull the headline BET fit results out of a parsed BET Report section.

    Parameters
    ----------
    section : dict or None
        Parsed ``"BET Report"`` section, or ``None`` if absent.

    Returns
    -------
    dict or None
        Clean dict of BET scalar results (surface area, fit slope/intercept,
        C constant, monolayer capacity, correlation coefficient, molecular
        cross-sectional area — with ``*_error``/``*_unit`` siblings where
        applicable), or ``None`` if ``section`` is ``None``.
    """
    if section is None:
        return None
    s = section["scalars"]
    return {
        "surface_area": _scalar_value(s, "BET surface area"),
        "surface_area_error": _scalar_error(s, "BET surface area"),
        "surface_area_unit": _scalar_unit(s, "BET surface area"),
        "slope": _scalar_value(s, "Slope"),
        "slope_error": _scalar_error(s, "Slope"),
        "y_intercept": _scalar_value(s, "Y-intercept"),
        "y_intercept_error": _scalar_error(s, "Y-intercept"),
        "c_constant": _scalar_value(s, "C"),
        "qm": _scalar_value(s, "Qm"),
        "qm_unit": _scalar_unit(s, "Qm"),
        "correlation_coefficient": _scalar_value(s, "Correlation coefficient"),
        "cross_sectional_area": _scalar_value(s, "Molecular cross-sectional area"),
        "cross_sectional_area_unit": _scalar_unit(s, "Molecular cross-sectional area"),
    }


def _extract_isotherm(section: Optional[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Split the Isotherm Tabular Report table into adsorption/desorption branches.

    The branch turnover is the row of maximum relative pressure; everything
    up to and including it is the adsorption branch, everything after is
    desorption. Rows missing a relative pressure or quantity-adsorbed value
    (e.g. the initial free-space dose) are dropped.

    Parameters
    ----------
    section : dict or None
        Parsed ``"Isotherm Tabular Report"`` section, or ``None`` if absent.

    Returns
    -------
    dict
        Mapping with keys ``"p_rel_ads"``, ``"q_ads"``, ``"p_rel_des"``,
        ``"q_des"`` (empty dict if ``section`` is ``None`` or has no table).
    """
    if section is None or section["table"] is None:
        return {}

    table = section["table"]
    df = pd.DataFrame(table["rows"], columns=table["columns"])
    p_rel = pd.to_numeric(df["Relative Pressure (P/Po)"], errors="coerce").to_numpy()
    q = pd.to_numeric(df["Quantity Adsorbed (cm\xb3/g STP)"], errors="coerce").to_numpy()

    valid = ~(np.isnan(p_rel) | np.isnan(q))
    p_rel, q = p_rel[valid], q[valid]
    if p_rel.size == 0:
        return {}

    split = int(np.argmax(p_rel)) + 1
    p_rel_ads, q_ads = p_rel[:split], q[:split]
    p_rel_des, q_des = p_rel[split:], q[split:]

    out: dict[str, np.ndarray] = {}
    if p_rel_ads.size:
        order = np.argsort(p_rel_ads)
        out["p_rel_ads"], out["q_ads"] = p_rel_ads[order], q_ads[order]
    if p_rel_des.size:
        order = np.argsort(p_rel_des)
        out["p_rel_des"], out["q_des"] = p_rel_des[order], q_des[order]
    return out
