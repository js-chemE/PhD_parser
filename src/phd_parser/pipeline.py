"""Pipelines: a named list of processing steps that applies to any data object and can be saved and replayed.

The idea is RamanSPy's ``preprocessing.Pipeline``: write the processing once
as a list of steps, run it on one spectrum or on a whole folder of them, and
keep it next to the results. Here a step is simply the name of a method of
the data class plus its keyword arguments, so a pipeline works for every
class of the package alike (``SpectroscopyData``, ``XPSData``, ``MSData``,
``LVData``, ``GCData``, ...), needs no wrapper per operation, and is plain
JSON: ``to_json`` / ``from_json`` store it with the processed data.

Because every method records itself in the object's history with its
arguments, a pipeline can also be *read off* a processed object:
``Pipeline.from_history(ir)`` gives the steps that made ``ir``, ready to be
applied to the next run. The history is the pipeline's natural counterpart:
one says what was done, the other does it again.

Example::

    from phd_parser import Pipeline

    drifts = Pipeline([
        ("select_tos_range", {"min_s": 0, "max_s": 6 * 3600}),
        ("correct_baseline", {"anchor_range_cm": (2600, 2500), "control_points_cm": [3450, 2720, 1750]}),
        ("select_wavenumber_range", {"min_cm": 1000, "max_cm": 3900}),
        "normalise_max",
    ], name="drifts standard")

    ir = drifts(ir_raw)                      # one object
    runs = drifts.map({"JV001": ir_1, "JV002": ir_2})   # several at once, keys kept
    drifts.to_json(folder / "pipeline.json")
"""

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field

StepLike = Union["Step", str, tuple, Mapping[str, Any]]


class Step(BaseModel):
    """One processing step: the name of a method of the data class and its keyword arguments."""

    model_config = ConfigDict(frozen=True)

    method: str = Field(description="Name of the method to call on the data object, e.g. 'smooth_savgol'.")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Keyword arguments of the call.")

    @classmethod
    def coerce(cls, step: StepLike) -> "Step":
        """Read a step from any of its spellings.

        Parameters
        ----------
        step : Step, str, tuple or dict
            A ``Step``; a bare method name (``"normalise_max"``); a
            ``(method, parameters)`` tuple; a ``{"method": ..., "parameters": ...}``
            dict (as written by :meth:`Pipeline.to_json`); or a one-entry dict
            ``{"smooth_savgol": {"window_length": 11}}``.

        Returns
        -------
        Step
            The step.

        Raises
        ------
        ValueError
            For anything else.
        """
        if isinstance(step, Step):
            return step
        if isinstance(step, str):
            return cls(method=step)
        if isinstance(step, tuple):
            if len(step) == 1:
                return cls(method=step[0])
            if len(step) == 2 and isinstance(step[1], Mapping):
                return cls(method=step[0], parameters=dict(step[1]))
            raise ValueError(f"A step tuple is (method, parameters), got {step!r}")
        if isinstance(step, Mapping):
            if "method" in step:
                return cls(method=step["method"], parameters=dict(step.get("parameters") or {}))
            if len(step) == 1:
                (method, parameters), = step.items()
                return cls(method=method, parameters=dict(parameters or {}))
        raise ValueError(f"Cannot read a step from {step!r}")

    def apply(self, data: Any) -> Any:
        """Call the step's method on ``data`` and return the result.

        Parameters
        ----------
        data : data object
            Anything with a method named :attr:`method`.

        Returns
        -------
        data object
            What the method returned, normally a new instance.

        Raises
        ------
        AttributeError
            If the object has no such method.
        """
        method = getattr(data, self.method, None)
        if method is None or not callable(method):
            raise AttributeError(f"{type(data).__name__} has no method {self.method!r} (pipeline step)")
        return method(**self.parameters)

    def __repr__(self) -> str:
        """``method(key=value, ...)``."""
        parameters = ", ".join(f"{key}={value!r}" for key, value in self.parameters.items())
        return f"{self.method}({parameters})"


class Pipeline(BaseModel):
    """An ordered list of :class:`Step` applied to a data object, one after the other.

    Immutable like the data classes: :meth:`append` and ``+`` return a new
    pipeline. Calling it applies it: ``pipeline(data)``.

    Attributes
    ----------
    steps : list of Step
        The steps, in order.
    name : str
        A label, kept in the JSON file.
    """

    model_config = ConfigDict(frozen=True)

    steps: list[Step] = Field(default_factory=list)
    name: str = ""

    def __init__(self, steps: Optional[Iterable[StepLike]] = None, /, *, name: str = "") -> None:
        """Build a pipeline from steps in any spelling :meth:`Step.coerce` reads.

        Parameters
        ----------
        steps : iterable of Step, str, tuple or dict, optional
            The steps in order.
        name : str, optional
            A label.
        """
        super().__init__(steps=[Step.coerce(step) for step in (steps or [])], name=name)

    # ---------------------------------------------------------------- running

    def apply(self, data: Any) -> Any:
        """Apply every step in order and return the result (the input is not changed).

        Parameters
        ----------
        data : data object
            A ``SpectroscopyData``, ``XPSData``, ``MSData``, ... anything
            whose methods the steps name.

        Returns
        -------
        data object
            The output of the last step; ``data`` itself for an empty pipeline.
        """
        for step in self.steps:
            data = step.apply(data)
        return data

    def __call__(self, data: Any) -> Any:
        """Same as :meth:`apply`."""
        return self.apply(data)

    def map(self, items: Union[Mapping[Any, Any], Sequence[Any]]) -> Union[dict[Any, Any], list[Any]]:
        """Apply the pipeline to every item of a dict or a sequence.

        Parameters
        ----------
        items : dict or sequence of data objects
            A dict keeps its keys (the regions of an XPS file, the runs of a
            series); a sequence gives a list.

        Returns
        -------
        dict or list
            The results in the same shape.
        """
        if isinstance(items, Mapping):
            return {key: self.apply(value) for key, value in items.items()}
        return [self.apply(item) for item in items]

    # ---------------------------------------------------------------- building

    def append(self, step: StepLike) -> "Pipeline":
        """Return a new pipeline with one more step at the end.

        Parameters
        ----------
        step : Step, str, tuple or dict
            In any spelling :meth:`Step.coerce` reads.

        Returns
        -------
        Pipeline
            The longer pipeline; ``self`` is unchanged.
        """
        return Pipeline([*self.steps, Step.coerce(step)], name=self.name)

    def __add__(self, other: "Pipeline") -> "Pipeline":
        """Concatenate two pipelines (the left one's name is kept)."""
        if not isinstance(other, Pipeline):
            return NotImplemented
        return Pipeline([*self.steps, *other.steps], name=self.name or other.name)

    def __len__(self) -> int:
        """Number of steps."""
        return len(self.steps)

    def __iter__(self) -> Iterator[Step]:  # type: ignore[override]
        """Iterate over the steps."""
        return iter(self.steps)

    def __getitem__(self, index: Union[int, slice]) -> Union[Step, "Pipeline"]:
        """A step by position, or a sub-pipeline for a slice."""
        if isinstance(index, slice):
            return Pipeline(self.steps[index], name=self.name)
        return self.steps[index]

    # ---------------------------------------------------------------- from a history

    @classmethod
    def from_history(
        cls, data: Any, name: str = "", include_constructor: bool = False, since: Optional[str] = None
    ) -> "Pipeline":
        """The pipeline that made ``data``, read off its processing history.

        Every recorded step becomes a :class:`Step` with the recorded
        arguments, so applying the result to another raw object repeats the
        processing. Constructors (``from_*``) are left out unless asked for,
        since they are not methods of an existing object.

        Parameters
        ----------
        data : data object
            Anything with a ``history`` (see :mod:`phd_parser.history`).
        name : str, optional
            Label of the new pipeline.
        include_constructor : bool, optional
            Keep the ``from_*`` entries as steps (default is ``False``).
        since : str or None, optional
            Start after the *last* entry with this step name, e.g.
            ``since="from_netcdf"`` for what one processing stage did to the
            file it loaded (default is ``None``, the whole history). A trailing
            entry of that name (the load that produced ``data`` itself) is not
            counted, so the stage of a freshly loaded file is read correctly.

        Returns
        -------
        Pipeline
            The replayable steps.

        Raises
        ------
        ValueError
            If a recorded argument cannot be replayed: a long array or a data
            object, which the history keeps only as a summary. The message
            names the step and the argument; rebuild that step by hand.
        """
        history = list(getattr(data, "history", None) or [])
        if since is not None:
            positions = [index for index, entry in enumerate(history) if entry.get("step") == since]
            if not positions:
                raise ValueError(f"No step {since!r} in the history; steps are {[entry.get('step') for entry in history]}")
            # the load that produced this very object sits at the end: it is the boundary of the
            # next stage, not of this one, so start after the occurrence before it
            if positions[-1] == len(history) - 1:
                positions = positions[:-1]
            history = history[positions[-1] + 1:] if positions else history
        steps = []
        for entry in history:
            method = str(entry.get("step", ""))
            if not include_constructor and (method.startswith("from_") or method.startswith("_from_")):
                continue
            parameters = dict(entry.get("parameters") or {})
            for key, value in parameters.items():
                if isinstance(value, dict) and ("array_shape" in value or ("sizes" in value and "type" in value)):
                    raise ValueError(
                        f"Step {method!r}: argument {key!r} was recorded as a summary ({value}) and cannot be "
                        "replayed from the history; add that step to the pipeline by hand."
                    )
            steps.append(Step(method=method, parameters=parameters))
        return cls(steps, name=name)

    # ---------------------------------------------------------------- JSON

    def to_json(self, path: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        """The pipeline as JSON text, optionally written to a file.

        Parameters
        ----------
        path : str or pathlib.Path or None, optional
            Where to write it; nothing is written for ``None``.
        indent : int, optional
            JSON indentation (default is 2).

        Returns
        -------
        str
            The JSON text: ``{"name": ..., "steps": [{"method": ..., "parameters": {...}}, ...]}``.
        """
        from phd_parser.history import to_json_safe

        payload = {"name": self.name, "steps": [{"method": step.method, "parameters": to_json_safe(step.parameters)}
                                                for step in self.steps]}
        text = json.dumps(payload, indent=indent)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, source: Union[str, Path]) -> "Pipeline":
        """Read a pipeline written by :meth:`to_json`.

        Parameters
        ----------
        source : str or pathlib.Path
            A path to a JSON file, or the JSON text itself.

        Returns
        -------
        Pipeline
            The pipeline.
        """
        text = source
        if isinstance(source, Path) or (isinstance(source, str) and not source.lstrip().startswith("{")):
            text = Path(source).read_text(encoding="utf-8")
        payload = json.loads(text)
        return cls(payload.get("steps", []), name=payload.get("name", ""))

    def describe(self) -> str:
        """The steps as numbered lines, the way ``format_history`` prints a history."""
        head = f"Pipeline {self.name!r}" if self.name else "Pipeline"
        body = "\n".join(f"{number}. {step!r}" for number, step in enumerate(self.steps, start=1))
        return head + (":\n" + body if body else " (no steps)")

    def __repr__(self) -> str:
        """``Pipeline(name, n steps)`` with the steps listed."""
        return self.describe()
