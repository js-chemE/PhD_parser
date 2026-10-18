"""Joining two measurements of one experiment along the scan axis across a spectrometer restart.

A mixin of :class:`~phd_parser.spectroscopy.core.SpectroscopyData`; never used on its own.
"""

import json
import logging
from typing import TYPE_CHECKING
from typing import Any, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from phd_parser.history import records_step
from phd_parser.spectroscopy._common import (
    BACKGROUND_REFERENCED_TYPES,
    _as_single_spectrum,
    _is_spectroscopy_data,
)

if TYPE_CHECKING:
    from phd_parser.spectroscopy.core import SpectroscopyData

logger = logging.getLogger(__name__)


class _MergeMixin:
    """Joining two measurements of one experiment along the scan axis across a spectrometer restart."""

    @records_step
    def merge(
        self,
        other: "SpectroscopyData",
        keep_background: Union[Literal["first", "last", "none"], npt.NDArray, "SpectroscopyData"] = "first",
        order: Literal["auto", "given"] = "auto",
        sort: bool = True,
        tos_offset_seconds: Optional[float] = None,
        on_overlap: Literal["warn", "raise", "ignore", "trim"] = "warn",
        wavenumber: Literal["strict", "interp"] = "strict",
        convert_to_single_beam: bool = True,
    ) -> "SpectroscopyData":
        """Combine two measurements into one along the scan axis.

        This is the operation needed when a run had to be interrupted, the
        spectrometer was restarted mid-experiment and a *second* background was
        recorded, and the two resulting files describe one continuous
        experiment.  Merging happens along the scan (time) axis; the wavenumber
        axis is untouched.

        Because a spectrum is only physically comparable across a restart in
        raw detector units, the merge itself happens on ``single_beam`` data.
        Anything else is handled for you: both segments are converted with
        :meth:`to_single_beam`, merged, and converted back to the original
        ``data_type`` against the one surviving background, which is exactly
        the physically correct thing, since it re-references the second segment
        to the background that survived.  Two segments that already share an
        identical background skip the round trip and are merged as-is.

        Decisions this method makes
        ---------------------------
        data_type
            Preserved.  Non-single-beam data is rebased through single beam
            (see above), so the second segment's values change: they are
            recomputed against the surviving background.  Converting through
            single beam drops any stored baseline, with a warning.
        order
            Segments are ordered chronologically (``order='auto'``), i.e. by
            the absolute time of their first scan, *not* by ``tos`` alone,
            two files each starting at ``tos=0`` are only comparable through
            their ``tos_start``.  The earlier segment is called *first* below.
        tos / tos_start
            The merged data keeps the ``tos_start`` of the *first* segment;
            its ``tos`` values are left untouched.  The later segment's ``tos``
            values are shifted by the difference between the two ``tos_start``
            values, so the whole merged run is expressed on one origin.
        background
            Only one background is kept, by default the *first* segment's,
            i.e. the one recorded before the experiment started, not the one
            recorded mid-experiment after the restart.
        baseline
            Kept only if *both* segments carry one; otherwise it is dropped
            with a warning.
        scan ids
            Renumbered ``0 … n-1`` over the merged data.
        attrs
            Taken from the first segment, filled up with keys only the second
            has.  A JSON record of the operation is appended to
            ``ds.attrs['merge_log']``.

        Parameters
        ----------
        other : SpectroscopyData
            The second measurement.  1-D operands are promoted to a single
            scan, so two single spectra merge into a 2-D instance.
        keep_background : {'first', 'last', 'none'} or numpy.ndarray or SpectroscopyData, optional
            Which background survives (default is ``'first'``).  ``'first'``
            and ``'last'`` refer to the chronological order, not to the call
            order.  An explicit array or 1-D ``SpectroscopyData`` (single_beam) overrides
            both.  If the chosen segment has no background but the other one
            does, the available one is used with a warning.
        order : {'auto', 'given'}, optional
            Which segment counts as *first*, it defines the ``tos`` origin,
            the surviving background and the leading block.  ``'auto'``
            (default) picks the chronologically earlier one; ``'given'`` keeps
            ``self`` first regardless of timestamps (its ``tos`` then stays
            untouched and the other segment's may become negative).
        sort : bool, optional
            Sort the merged scans by ``tos`` (default is ``True``).  This is
            independent of ``order``: with ``order='given'`` and ``sort=True``
            the blocks are still re-ordered in time, only the origin changes.
            Pass ``sort=False`` to keep the two segments as contiguous blocks.
        tos_offset_seconds : float or None, optional
            Explicit shift in seconds applied to ``other``'s ``tos`` values to
            express them on ``self``'s time axis.  Overrides the shift derived
            from ``tos_start`` and is the way to merge segments that have no
            absolute timestamps (default is ``None``).
        on_overlap : {'warn', 'raise', 'ignore', 'trim'}, optional
            What to do when the later segment starts before the earlier one
            ends: warn (default), raise, accept silently, or drop the
            overlapping scans of the later segment.
        wavenumber : {'strict', 'interp'}, optional
            ``'strict'`` (default) requires identical wavenumber axes.
            ``'interp'`` interpolates the second segment onto the first
            segment's grid, restricted to the range covered by both.
        convert_to_single_beam : bool, optional
            Rebase non-single-beam segments through single beam automatically
            (default is ``True``).  Set to ``False`` to raise instead and keep
            the conversion in your own hands.  Ignored when both segments
            already share an identical background.

        Returns
        -------
        SpectroscopyData
            New 2-D instance holding the scans of both measurements, in the
            same ``data_type`` as the operands.

        Raises
        ------
        TypeError
            If ``other`` is not an ``SpectroscopyData``.
        ValueError
            If the two ``data_type`` values differ; if the data are not
            ``single_beam``, do not share a background, and either
            ``convert_to_single_beam=False`` or a segment has no background of
            its own to convert with; if the wavenumber axes are incompatible;
            if only one segment carries a ``tos_start`` and no
            ``tos_offset_seconds`` is given; or if ``on_overlap='raise'`` and
            the segments overlap in time.
        """
        if not _is_spectroscopy_data(other):
            raise TypeError(f"'other' must be a SpectroscopyData, got {type(other)}")
        if self.is_map or other.is_map:
            raise ValueError("merge joins scans along the time axis; a map has no scan axis to merge along")
        if self.data_type != other.data_type:
            raise ValueError(
                f"Cannot merge SpectroscopyData with different data_type "
                f"('{self.data_type}' and '{other.data_type}')"
            )

        ds_a, ds_b, wavenumber_si = self._align_wavenumber_for_merge(other, wavenumber)
        seg_a = self._merge_segment(ds_a)
        seg_b = self._merge_segment(ds_b)

        rebasing = "not_applicable"
        if self.data_type in BACKGROUND_REFERENCED_TYPES:
            bg_a, bg_b = seg_a["background"], seg_b["background"]
            if bg_a is None or bg_b is None:
                # Nothing to rebase with: OMNIC-style exports often carry only the
                # finished absorbance. Merging as recorded is the only option.
                rebasing = "none"
                missing = (
                    "neither segment carries" if bg_a is None and bg_b is None
                    else "one of the segments does not carry"
                )
                logger.warning(
                    f"Merging '{self.data_type}' data but {missing} a background, so the two "
                    "segments cannot be put on a common basis: they are merged exactly as "
                    "recorded, and the second segment stays referenced to whatever background it "
                    "was measured against. Assign the backgrounds with .with_background() on each "
                    "segment before merging to have them rebased properly."
                )
            elif np.allclose(bg_a, bg_b):
                rebasing = "shared_background"
                logger.info(
                    f"Merging '{self.data_type}' data as-is: both segments share the same background."
                )
            elif convert_to_single_beam:
                return self._merge_via_single_beam(
                    other,
                    keep_background=keep_background,
                    order=order,
                    sort=sort,
                    tos_offset_seconds=tos_offset_seconds,
                    on_overlap=on_overlap,
                    wavenumber=wavenumber,
                )
            else:
                raise ValueError(
                    f"Merging is defined on 'single_beam' data, got '{self.data_type}', and the two "
                    "segments were recorded against different backgrounds. Drop "
                    "convert_to_single_beam=False to let merge rebase them through single beam "
                    "automatically, or do it by hand with .to_single_beam() on both."
                )

        # Put both segments on one time axis, then decide which one came first.
        has_tos = seg_a["tos"] is not None and seg_b["tos"] is not None
        if has_tos:
            offset_b = self._merge_tos_offset(seg_a, seg_b, tos_offset_seconds)
            tos_a, tos_b = seg_a["tos"], seg_b["tos"] + offset_b
            a_first = True if order == "given" else float(tos_a.min()) <= float(tos_b.min())
            if not a_first:
                # Re-express on the other segment's origin so the first segment keeps its own tos.
                tos_a, tos_b = seg_a["tos"] - offset_b, seg_b["tos"]
        else:
            if tos_offset_seconds is not None:
                raise ValueError(
                    "tos_offset_seconds requires both segments to carry a 'tos' coordinate"
                )
            logger.warning(
                "At least one segment has no 'tos' coordinate: merging in the given order "
                "and the merged data will have no 'tos' coordinate either."
            )
            a_first = True
            tos_a = tos_b = None
            offset_b = None

        first, second = (seg_a, seg_b) if a_first else (seg_b, seg_a)
        tos_first, tos_second = (tos_a, tos_b) if a_first else (tos_b, tos_a)
        values_first, values_second = first["values"], second["values"]
        baseline_first, baseline_second = first["baseline"], second["baseline"]

        # Overlap in time: a restart normally leaves a gap, an overlap means
        # the two files describe (partly) the same scans.
        n_trimmed = 0
        if has_tos and float(tos_second.min()) <= float(tos_first.max()) and on_overlap != "ignore":
            overlap = float(tos_first.max()) - float(tos_second.min())
            message = (
                f"The two segments overlap in time by {overlap:.1f}s: the second segment starts at "
                f"tos={float(tos_second.min()):.1f}s while the first one runs until "
                f"tos={float(tos_first.max()):.1f}s"
            )
            if on_overlap == "raise":
                raise ValueError(message + " (on_overlap='raise')")
            if on_overlap == "trim":
                keep = tos_second > float(tos_first.max())
                n_trimmed = int((~keep).sum())
                if not keep.any():
                    raise ValueError(
                        message + "; trimming would leave no scan of the second segment. "
                        "Pass on_overlap='warn' or 'ignore' to keep both segments as they are."
                    )
                tos_second = tos_second[keep]
                values_second = values_second[keep]
                if baseline_second is not None:
                    baseline_second = baseline_second[keep]
                logger.warning(f"{message}; dropped {n_trimmed} overlapping scans of the second segment.")
            else:
                logger.warning(f"{message}. Pass on_overlap='trim' to drop the duplicated scans.")

        # Report where the two segments actually meet, so the join can be checked
        # against what happened in the lab rather than taken on trust.
        gap_seconds: Optional[float] = None
        if has_tos:
            gap_seconds = float(tos_second.min()) - float(tos_first.max())
            if first["tos_start"] is not None:
                end_of_first = first["tos_start"] + pd.Timedelta(seconds=float(tos_first.max()))
                start_of_second = first["tos_start"] + pd.Timedelta(seconds=float(tos_second.min()))
                logger.info(
                    f"Joining segments: first ends {end_of_first}, second starts {start_of_second} "
                    f"(gap {gap_seconds:.1f}s)."
                )
            else:
                logger.info(
                    f"Joining segments: first ends at tos={float(tos_first.max()):.1f}s, second "
                    f"starts at tos={float(tos_second.min()):.1f}s (gap {gap_seconds:.1f}s)."
                )

        values = np.vstack([values_first, values_second])
        tos = np.concatenate([tos_first, tos_second]) if has_tos else None

        has_both_baselines = baseline_first is not None and baseline_second is not None
        if not has_both_baselines and (baseline_first is not None or baseline_second is not None):
            logger.warning(
                "Only one of the two segments carries a baseline; dropping it from the merged data. "
                "Re-run baseline correction on the merged data if needed."
            )
        baseline = np.vstack([baseline_first, baseline_second]) if has_both_baselines else None

        if sort and has_tos:
            sort_idx = np.argsort(tos, kind="stable")
            values, tos = values[sort_idx], tos[sort_idx]
            if baseline is not None:
                baseline = baseline[sort_idx]

        background, background_attrs, background_source = self._resolve_merge_background(
            keep_background, first, second, wavenumber_si
        )

        attrs = {**second["attrs"], **first["attrs"]}
        if first["tos_start"] is not None:
            attrs["tos_start"] = first["tos_start"].isoformat()
        else:
            attrs.pop("tos_start", None)
        attrs["merge_log"] = json.dumps(
            [*first["merge_log"], *second["merge_log"], {
                "n_scans_first": int(values_first.shape[0]),
                "n_scans_second": int(values_second.shape[0]),
                "tos_start_first": first["tos_start"].isoformat() if first["tos_start"] else None,
                "tos_start_second": second["tos_start"].isoformat() if second["tos_start"] else None,
                "tos_offset_seconds": float(abs(offset_b)) if offset_b is not None else None,
                "background_kept": background_source,
                "wavenumber": wavenumber,
                "sorted": bool(sort and has_tos),
                "n_scans_trimmed": n_trimmed,
                "gap_seconds": gap_seconds,
                "rebasing": rebasing,
            }]
        )

        ds_new = self._build_ds(wavenumber_si, values, tos=tos, attrs=attrs, sort_by_tos=False)  # merge sorted already, or was asked not to
        if background is not None:
            ds_new = ds_new.assign({
                "background": xr.DataArray(
                    data=background,
                    coords={"wavenumber": ds_new.coords["wavenumber"]},
                    dims=["wavenumber"],
                    name="background",
                    attrs=background_attrs or {"data_type": "single_beam"},
                )
            })
        ds_new = self._with_baseline(ds_new, baseline)
        return self._new(ds=ds_new)

    @classmethod
    def merge_all(cls, items: Sequence["SpectroscopyData"], **merge_kwargs: Any) -> "SpectroscopyData":
        """Merge any number of measurements into one along the scan axis.

        Folds :meth:`merge` over ``items`` from left to right, so all its
        decisions (chronological ordering, one surviving background, shared
        ``tos`` origin) apply to every step.

        Parameters
        ----------
        items : sequence of SpectroscopyData
            Measurements to merge.  A sequence of length one is returned
            unchanged.
        **merge_kwargs
            Keyword arguments forwarded verbatim to :meth:`merge`.

        Returns
        -------
        SpectroscopyData
            New instance holding the scans of all inputs.

        Raises
        ------
        ValueError
            If ``items`` is empty.
        """
        items = list(items)
        if not items:
            raise ValueError("merge_all requires at least one SpectroscopyData")
        merged = items[0]
        for item in items[1:]:
            merged = merged.merge(item, **merge_kwargs)
        return merged

    def _merge_via_single_beam(self, other: "SpectroscopyData", **merge_kwargs: Any) -> "SpectroscopyData":
        """Merge two background-dependent segments by rebasing them through single beam."""
        source_data_type = self.data_type
        logger.info(
            f"Merging '{source_data_type}' data recorded against different backgrounds: converting "
            "both segments to single_beam, merging, then converting back against the surviving "
            "background."
        )

        try:
            left, right = self.to_single_beam(), other.to_single_beam()
        except ValueError as exc:
            raise ValueError(
                f"Cannot merge '{source_data_type}' data recorded against different backgrounds: "
                f"{exc} Each segment needs its own background so it can be put back into raw "
                "detector units before merging."
            ) from exc

        merged = left.merge(right, **merge_kwargs)

        if not merged.has_background:
            logger.warning(
                f"The merged data has no background (keep_background='none'), so it stays "
                f"single_beam instead of returning to '{source_data_type}'."
            )
            restored = merged
        else:
            restored = merged._to_data_type(source_data_type)

        return restored._note_in_merge_log(
            converted_via_single_beam=True,
            rebasing="converted",
            data_type=restored.data_type,
        )

    def _note_in_merge_log(self, **fields: Any) -> "SpectroscopyData":
        """Add fields to the most recent merge_log entry."""
        raw = self.ds.attrs.get("merge_log")
        if not raw:
            return self
        log = json.loads(raw)
        log[-1].update(fields)
        ds = self.ds.copy()
        ds.attrs = {**self.ds.attrs, "merge_log": json.dumps(log)}
        return self._new(ds=ds)

    def _align_wavenumber_for_merge(
        self, other: "SpectroscopyData", mode: Literal["strict", "interp"]
    ) -> Tuple[xr.Dataset, xr.Dataset, npt.NDArray]:
        """Bring both datasets onto one wavenumber grid before merging."""
        wn_self, wn_other = self.wavenumber, other.wavenumber
        if wn_self.shape == wn_other.shape and np.allclose(wn_self, wn_other):
            return self.ds, other.ds, wn_self

        if mode == "strict":
            raise ValueError(
                f"Cannot merge SpectroscopyData with different wavenumber axes: "
                f"{wn_self.size} points over {wn_self.min() / 100:.1f}–{wn_self.max() / 100:.1f} cm⁻¹ "
                f"vs {wn_other.size} points over {wn_other.min() / 100:.1f}–{wn_other.max() / 100:.1f} cm⁻¹. "
                "Pass wavenumber='interp' to interpolate the second segment onto the first's grid."
            )
        if mode != "interp":
            raise ValueError(f"wavenumber must be 'strict' or 'interp', got {mode!r}")

        lo = max(float(wn_self.min()), float(wn_other.min()))
        hi = min(float(wn_self.max()), float(wn_other.max()))
        if lo >= hi:
            raise ValueError(
                f"The wavenumber ranges of the two segments do not overlap "
                f"({wn_self.min() / 100:.1f}–{wn_self.max() / 100:.1f} cm⁻¹ vs "
                f"{wn_other.min() / 100:.1f}–{wn_other.max() / 100:.1f} cm⁻¹)"
            )

        grid = wn_self[(wn_self >= lo) & (wn_self <= hi)]
        ds_self = self.ds.sel(wavenumber=(wn_self >= lo) & (wn_self <= hi))
        # interp needs a monotonically increasing source axis
        ds_other = other.ds.sortby("wavenumber").interp(wavenumber=grid)
        ds_other.attrs = dict(other.ds.attrs)
        logger.warning(
            f"Merging onto the first segment's wavenumber grid restricted to the common range "
            f"{grid.min() / 100:.1f}–{grid.max() / 100:.1f} cm⁻¹ ({grid.size} points); "
            "the second segment was interpolated onto it."
        )
        return ds_self, ds_other, grid

    @staticmethod
    def _merge_tos_offset(
        seg_self: dict[str, Any],
        seg_other: dict[str, Any],
        tos_offset_seconds: Optional[float],
    ) -> float:
        """Seconds to add to the second segment's tos to express it on the first's time axis."""
        if tos_offset_seconds is not None:
            return float(tos_offset_seconds)

        tos_start_self, tos_start_other = seg_self["tos_start"], seg_other["tos_start"]
        if tos_start_self is not None and tos_start_other is not None:
            return float((tos_start_other - tos_start_self).total_seconds())
        if tos_start_self is None and tos_start_other is None:
            logger.warning(
                "Neither segment has a 'tos_start': assuming both 'tos' axes share the same origin. "
                "Pass tos_offset_seconds=... or set_tos_start(...) if they do not."
            )
            return 0.0
        raise ValueError(
            "Only one of the two segments has a 'tos_start', so their time axes cannot be related. "
            "Set the missing one with .set_tos_start(...) or pass tos_offset_seconds=..."
        )

    @staticmethod
    def _resolve_merge_background(
        keep_background: Union[Literal["first", "last", "none"], npt.NDArray, "SpectroscopyData"],
        first: dict[str, Any],
        second: dict[str, Any],
        wavenumber_si: npt.NDArray,
    ) -> Tuple[Optional[npt.NDArray], Optional[dict[str, Any]], str]:
        """Pick the single background the merged data keeps."""
        if _is_spectroscopy_data(keep_background):
            if keep_background.data_type not in (None, "single_beam"):
                raise ValueError(
                    f"Background SpectroscopyData must be single_beam, got '{keep_background.data_type}'"
                )
            values = _as_single_spectrum(keep_background)
            wn = keep_background.wavenumber
            if wn.shape != wavenumber_si.shape or not np.allclose(wn, wavenumber_si):
                sort_idx = np.argsort(wn)
                values = np.interp(wavenumber_si, wn[sort_idx], values[sort_idx])
                logger.warning("Interpolated the supplied background onto the merged wavenumber grid.")
            return values, {"data_type": "single_beam"}, "explicit"

        if not isinstance(keep_background, str):
            values = np.asarray(keep_background, dtype=float)
            if values.ndim != 1:
                raise ValueError("Background array must be 1-D")
            if values.size != wavenumber_si.size:
                raise ValueError(
                    f"Background size ({values.size}) does not match the merged wavenumber axis "
                    f"({wavenumber_si.size})"
                )
            return values, {"data_type": "single_beam"}, "explicit"

        if keep_background == "none":
            return None, None, "none"
        if keep_background not in ("first", "last"):
            raise ValueError(
                "keep_background must be 'first', 'last', 'none', an array or an SpectroscopyData, "
                f"got {keep_background!r}"
            )

        chosen, fallback = (first, second) if keep_background == "first" else (second, first)
        source = keep_background
        if chosen["background"] is None and fallback["background"] is not None:
            source = "last" if keep_background == "first" else "first"
            logger.warning(
                f"keep_background='{keep_background}' but that segment carries no background; "
                f"keeping the '{source}' segment's background instead."
            )
            chosen = fallback
        if chosen["background"] is None:
            return None, None, "none"
        return chosen["background"], chosen["background_attrs"], source

    @classmethod
    def _merge_segment(cls, ds: xr.Dataset) -> dict[str, Any]:
        """Flatten one merge operand into plain arrays (1-D data becomes a single scan)."""
        data = ds["data"]
        is_1d = data.ndim == 1

        if "tos" in ds.coords:
            tos = np.atleast_1d(np.asarray(ds.coords["tos"].values, dtype=float))
        elif is_1d:
            # A lone spectrum sits at its own tos_start.
            tos = np.zeros(1, dtype=float)
        else:
            tos = None

        raw_tos_start = ds.attrs.get("tos_start")
        raw_merge_log = ds.attrs.get("merge_log")

        return {
            "values": data.values[None, :] if is_1d else data.values,
            "tos": tos,
            "tos_start": pd.Timestamp(raw_tos_start) if raw_tos_start is not None else None,
            "background": ds["background"].values if "background" in ds else None,
            "background_attrs": dict(ds["background"].attrs) if "background" in ds else None,
            "baseline": (
                (ds["baseline"].values[None, :] if is_1d else ds["baseline"].values)
                if "baseline" in ds
                else None
            ),
            "attrs": {k: v for k, v in ds.attrs.items() if k != "merge_log"},
            "merge_log": json.loads(raw_merge_log) if raw_merge_log else [],
        }
