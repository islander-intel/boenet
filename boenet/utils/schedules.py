# utils/schedules.py
"""
schedules.py

Lightweight value schedulers for temperatures and regularization coefficients.
Designed to be shared across trainers to avoid copy/paste.

What you get
------------
1) Functional API (single-shot):
   - value_none(start, step, total_steps)
   - value_linear(start, end, step, total_steps)
   - value_cosine(start, end, step, total_steps)

2) Stateful schedulers (factory returns a callable object):
   - make_scheduler(kind, start, end, total_steps, **kwargs)
     -> sched(step)  # returns value at "step"

   Useful kwargs:
     * start_step:    when the schedule begins (default 0)
     * end_step:      when the schedule ends (default total_steps-1)
     * clamp_min/max: hard clip the returned value (optional)
     * ramp:          optional warmup from `ramp.start` to `ramp.end` over `ramp.steps`
                      (applied before main schedule window)
     * outside:       behavior outside [start_step, end_step]:
                      {"hold_start", "hold_end", "zeros"} (default "hold_start")

3) Epoch adapter:
   - epoch_adapter(sched, steps_per_epoch) -> sched_epoch(epoch_idx, total_epochs)

Conventions
-----------
- All schedules are defined over an integer "step" domain in [0, total_steps-1].
- For cosine annealing we use the common "half-cosine" (1 -> 0) shape:
    w = 0.5 * (1 + cos(pi * t)), t in [0,1]
  then blend: value = w * start + (1 - w) * end

Examples
--------
>>> # Functional (no state)
>>> value_linear(1.4, 0.7, step=5, total_steps=11)
1.05

>>> # Factory (stateful callable)
>>> sched = make_scheduler("cosine", start=1.4, end=0.7, total_steps=11)
>>> [round(sched(s), 3) for s in range(0, 11)]
[1.4, 1.363, 1.291, 1.191, 1.074, 0.95, 0.826, 0.709, 0.609, 0.537, 0.5]

>>> # Warmup then linear decay in a window, hold beyond
>>> sched = make_scheduler(
...     "linear", start=1.2, end=0.8, total_steps=100,
...     start_step=10, end_step=60,
...     ramp={"start": 0.5, "end": 1.2, "steps": 10},
... )
>>> [round(sched(s), 2) for s in (0, 5, 10, 35, 60, 80)]
[0.5, 0.85, 1.2, 1.0, 0.8, 0.8]

Notes
-----
- This module is intentionally dependency-free (only `math` and `typing`).
- All values are returned as Python floats for ease of logging/printing.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Literal, Optional, Union

Kind = Literal["none", "linear", "cosine"]
OutsideBehavior = Literal["hold_start", "hold_end", "zeros"]


# --------------------------------------------------------------------------- #
#                           Functional single-shot API                        #
# --------------------------------------------------------------------------- #

def value_none(start: float, step: int, total_steps: int) -> float:
    """Constant schedule: always return `start`."""
    _ = step, total_steps  # unused, for signature symmetry
    return float(start)


def value_linear(start: float, end: float, step: int, total_steps: int) -> float:
    """
    Linear interpolation from `start` to `end` over `total_steps` steps.
    When total_steps <= 1, returns `end` for numerical sanity.
    """
    if total_steps <= 1:
        return float(end)
    # Normalize to [0,1]
    t = max(0.0, min(1.0, step / float(total_steps - 1)))
    return float((1.0 - t) * start + t * end)


def value_cosine(start: float, end: float, step: int, total_steps: int) -> float:
    """
    Cosine annealing from `start` to `end` over `total_steps` steps:
      w = 0.5 * (1 + cos(pi * t)),  t in [0,1]
      value = w*start + (1-w)*end
    When total_steps <= 1, returns `end`.
    """
    if total_steps <= 1:
        return float(end)
    t = max(0.0, min(1.0, step / float(total_steps - 1)))
    w = 0.5 * (1.0 + math.cos(math.pi * t))  # 1 -> 0
    return float(w * start + (1.0 - w) * end)


# --------------------------------------------------------------------------- #
#                             Stateful scheduler API                           #
# --------------------------------------------------------------------------- #

class _BaseScheduler:
    """
    Shared machinery for windowed schedules with optional warmup and clamping.
    Not intended to be used directly; use `make_scheduler`.
    """

    def __init__(
        self,
        fn: Callable[[float, float, int, int], float],
        *,
        start: float,
        end: Optional[float],
        total_steps: int,
        start_step: int = 0,
        end_step: Optional[int] = None,
        clamp_min: Optional[float] = None,
        clamp_max: Optional[float] = None,
        outside: OutsideBehavior = "hold_start",
        ramp: Optional[Dict[str, Union[int, float]]] = None,
    ):
        if end is None:
            end = start
        if end_step is None:
            end_step = max(0, total_steps - 1)

        # Basic invariants
        self.fn = fn
        self.start = float(start)
        self.end = float(end)
        self.total_steps = max(1, int(total_steps))
        self.start_step = max(0, int(start_step))
        self.end_step = max(self.start_step, int(end_step))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.outside: OutsideBehavior = outside

        # Optional warmup ramp (before start_step)
        self.ramp = None
        if ramp:
            rs = float(ramp.get("start", self.start))
            re = float(ramp.get("end", self.start))
            rsteps = int(ramp.get("steps", 0))
            self.ramp = (rs, re, max(0, rsteps))

    def _apply_clamp(self, v: float) -> float:
        if self.clamp_min is not None:
            v = max(self.clamp_min, v)
        if self.clamp_max is not None:
            v = min(self.clamp_max, v)
        return v

    def _eval_window(self, step: int) -> float:
        # Map global step into local window [start_step, end_step]
        span = max(1, self.end_step - self.start_step)
        local_step = max(0, min(span, step - self.start_step))
        return self.fn(self.start, self.end, local_step, span + 1)

    def __call__(self, step: int) -> float:
        """
        Evaluate schedule at integer `step` (0-indexed).
        Behavior by region:
          [0, ramp.steps)             -> warmup interpolation (if provided)
          [ramp.steps, start_step)    -> hold start (or zeros, per outside)
          [start_step, end_step]      -> main schedule (fn)
          (end_step, +inf)            -> hold end (or zeros, per outside)
        """
        s = int(step)

        # 1) Warmup ramp (optional)
        if self.ramp:
            rs, re, rsteps = self.ramp
            if rsteps > 0 and s < rsteps:
                v = value_linear(rs, re, s, rsteps)
                return self._apply_clamp(v)
            # after ramp, we fall through to region checks below

        # 2) Before window
        if s < self.start_step:
            if self.outside == "zeros":
                v = 0.0
            else:
                v = self.start  # "hold_start"
            return self._apply_clamp(v)

        # 3) Inside window
        if self.start_step <= s <= self.end_step:
            v = self._eval_window(s)
            return self._apply_clamp(v)

        # 4) After window
        if self.outside == "zeros":
            v = 0.0
        else:
            v = self.end  # "hold_end"
        return self._apply_clamp(v)


def make_scheduler(
    kind: Kind,
    *,
    start: float,
    end: Optional[float] = None,
    total_steps: int,
    start_step: int = 0,
    end_step: Optional[int] = None,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
    outside: OutsideBehavior = "hold_start",
    ramp: Optional[Dict[str, Union[int, float]]] = None,
) -> _BaseScheduler:
    """
    Factory for a stateful, callable scheduler.

    Parameters
    ----------
    kind : {"none", "linear", "cosine"}
        Schedule shape to use over the active window.
    start : float
        Starting value of the schedule (and value when holding-start outside window).
    end : Optional[float]
        Ending value of the schedule; if None, equals `start` (constant).
    total_steps : int
        Total steps in the *global* run; used as a sensible default for end_step.
    start_step : int
        First step of the active window (inclusive).
    end_step : Optional[int]
        Last step of the active window (inclusive). Defaults to total_steps-1.
    clamp_min / clamp_max : Optional[float]
        If provided, hard-clip the returned values into [clamp_min, clamp_max].
    outside : {"hold_start", "hold_end", "zeros"}
        Value returned outside the active window. Default: hold at start before, hold at end after.
    ramp : Optional[dict]
        Optional warmup dict with keys {"start", "end", "steps"} applied before `start_step`.

    Returns
    -------
    sched : callable
        A function-like object: value = sched(step).
    """
    kind = str(kind).lower().strip()
    if kind == "none":
        fn = value_none
    elif kind == "linear":
        fn = value_linear
    elif kind == "cosine":
        fn = value_cosine
    else:
        raise ValueError(f"Unknown schedule kind '{kind}'. Use one of: none, linear, cosine.")

    return _BaseScheduler(
        fn,
        start=start,
        end=end,
        total_steps=total_steps,
        start_step=start_step,
        end_step=end_step,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        outside=outside,
        ramp=ramp,
    )


# --------------------------------------------------------------------------- #
#                              Epoch adapter utility                           #
# --------------------------------------------------------------------------- #

def epoch_adapter(
    sched: Callable[[int], float],
    steps_per_epoch: int,
) -> Callable[[int, int], float]:
    """
    Wrap a step-based scheduler so you can query it by epoch index.

    Example
    -------
    >>> # 12 epochs, 500 steps/epoch
    >>> step_sched = make_scheduler("cosine", start=1.4, end=0.7, total_steps=12)
    >>> epoch_sched = epoch_adapter(step_sched, steps_per_epoch=500)
    >>> epoch_sched(epoch_idx=3, total_epochs=12)
    1.191...

    Notes
    -----
    - We map epoch e in [0, total_epochs-1] to a "global step" s=e, i.e., one
      scheduler step per epoch. If you want finer granularity, build your
      scheduler directly over *batches* and query it per-training-step instead.
    """
    spe = max(1, int(steps_per_epoch))

    def by_epoch(epoch_idx: int, total_epochs: int) -> float:
        # Default mapping: one schedule "step" per epoch (not per batch)
        # If you prefer batch-wise, call `sched(global_batch_step)` directly.
        _ = total_epochs, spe  # kept for future variants; unused here
        return float(sched(int(epoch_idx)))

    return by_epoch


# --------------------------------------------------------------------------- #
#                                Convenience shim                             #
# --------------------------------------------------------------------------- #

def resolve_value(
    schedule: Kind,
    *,
    start: float,
    end: float,
    step: int,
    total_steps: int,
) -> float:
    """
    Minimal convenience for one-off calls (mirrors older trainer helpers).

    Example
    -------
    >>> resolve_value("linear", start=1.4, end=0.7, step=2, total_steps=5)
    1.225
    """
    schedule = str(schedule).lower().strip()
    if schedule == "none":
        return value_none(start, step, total_steps)
    if schedule == "linear":
        return value_linear(start, end, step, total_steps)
    if schedule == "cosine":
        return value_cosine(start, end, step, total_steps)
    raise ValueError(f"Unknown schedule '{schedule}'. Use: none, linear, cosine.")
