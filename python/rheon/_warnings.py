"""The warnings of the library's promise: a name on its way out, and a surface
that is not yet promised. Nothing here imports the rest of the package, so
every module can use it.
"""

from __future__ import annotations

import functools
import os
import sys
import threading
import warnings
from collections.abc import Callable
from types import FrameType

from rheon import _env

#: The environment variable that turns :class:`UnstableWarning` on, as
#: ``POLARS_WARN_UNSTABLE`` does Polars' own: ``1`` and nothing else.
UNSTABLE_VAR = "RHEON_WARN_UNSTABLE"


class RheonWarning(Warning):
    """The base class of every warning this library issues.

    Filter on it to silence or surface every warning of this library at
    once, as Polars' ``PolarsWarning`` covers Polars' own. Each warning keeps
    its standard category as a second parent, so a filter on that category
    goes on working:

    .. list-table::
       :header-rows: 1
       :widths: 55 45

       * - the warning
         - its standard category
       * - :class:`~rheon.ReadinessWarning`,
           :class:`~rheon.polars.OrderNotGuaranteedWarning`,
           :class:`~rheon.polars.ConsumedSourceWarning` and
           :class:`~rheon.UnstableWarning`
         - ``UserWarning``
       * - :class:`~rheon.RheonDeprecationWarning`
         - ``DeprecationWarning``

    .. rubric:: Examples

    .. code-block:: python

        import warnings

        warnings.simplefilter("ignore", rp.RheonWarning)   # every one of them
        print(issubclass(rp.ReadinessWarning, rp.RheonWarning))

    .. code-block:: text

        True
    """


class RheonDeprecationWarning(RheonWarning, DeprecationWarning):
    """A warning that a name still read is on its way out.

    The library issues it for a name it still reads under a spelling it will
    stop reading at the next major version. The one it reads that way is an
    environment variable under its name from before the rename
    (``POLARS_ONLINE_MAX_THREADS`` for ``RHEON_MAX_THREADS``, and the same for
    ``RHEON_TIMING``, ``RHEON_WARN_UNSTABLE`` and ``RHEON_SKIP_CPU_CHECK``): it
    is read while the new name is not set, with this warning once at
    ``import rheon``, and not at all from the next major version. Any other
    renamed name is refused outright, naming the new one (``TypeError`` from a
    builder or a function, ``ValueError`` from a spec dict).
    Python shows a ``DeprecationWarning`` issued from a library only in
    ``__main__`` and under test runners, so to see it everywhere, filter it
    with ``warnings.simplefilter("default", rheon.RheonDeprecationWarning)``,
    and to make it an error, with ``"error"``.

    .. rubric:: Examples

    .. code-block:: python

        import warnings

        warnings.simplefilter("error", rp.RheonDeprecationWarning)   # an error from here
        print(issubclass(rp.RheonDeprecationWarning, DeprecationWarning))

    .. code-block:: text

        True

    .. rubric:: See also

    - :class:`RheonWarning`, the base class every warning of this
      library shares.
    """


class UnstableWarning(RheonWarning, UserWarning):
    """A warning that a part of the API not yet promised to stay the same was used.

    It is issued only when the environment variable
    ``RHEON_WARN_UNSTABLE`` is ``1``, as Polars issues its own
    ``UnstableWarning`` under ``POLARS_WARN_UNSTABLE``. Once
    ``rheon.polars`` is imported, the warning issued is
    ``rp.UnstableWarning``, a subclass of this class and of Polars' class,
    so a filter on either, or on ``polars.exceptions.UnstableWarning``,
    covers this library's unstable parts.

    Each unstable part says so in its docstring, and may change in any
    release without that counting as a breaking change:

    - the state file of :func:`rheon.polars.stream.with_windows`;
    - the written form of a formula target's tree, in a TOML file or a
      saved state;
    - :meth:`rheon.polars.ModelBank.fit_predict_arrow`,
      :meth:`rheon.polars.ModelBank.predict_arrow` and
      :class:`rheon.ArrowStruct`;
    - the modules :mod:`rheon.polars.sim`, :mod:`rheon.corr` and
      :mod:`rheon.polars._gbdt`.

    Everything else in ``__all__`` is promised.

    .. rubric:: Examples

    .. code-block:: python

        print(issubclass(rheon.UnstableWarning, rheon.RheonWarning))

    .. code-block:: text

        True

    .. rubric:: See also

    - :class:`RheonWarning`, the base class every warning of this
      library shares.
    """


#: The module that holds the Polars-parented :class:`UnstableWarning`, once
#: `rheon.polars` has imported it.
_POLARS_WARNINGS = "rheon.polars._warnings"


def _unstable_class() -> type[UnstableWarning]:
    """The :class:`UnstableWarning` to issue: ``rheon.polars``'s, a Polars
    ``UnstableWarning`` too, once that is imported, and this module's
    otherwise."""
    polars_warnings = sys.modules.get(_POLARS_WARNINGS)
    if polars_warnings is None:
        return UnstableWarning
    cls: type[UnstableWarning] = polars_warnings.UnstableWarning
    return cls


def _unstable_on() -> bool:
    return _env.get(UNSTABLE_VAR) == "1"


def warn_unstable(what: str, *, stacklevel: int = 3) -> None:
    """An :class:`UnstableWarning` about ``what``, when the variable is ``1``.
    ``stacklevel`` counts from the caller of this function: 3 is the caller's
    caller, the user's line under a wrapper."""
    if not _unstable_on():
        return
    warnings.warn(
        f"{what}: this is unstable, so it may change in any release without that counting "
        f"as a breaking change. Set {UNSTABLE_VAR}=0, or unset it, to silence this warning.",
        _unstable_class(),
        stacklevel=stacklevel,
    )


#: How deep this thread is in calls to functions labelled unstable, so that
#: only the outermost warns: one of a module's functions calling another is
#: one call of the user's.
_inside = threading.local()


def unstable[**P, R](what: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """A function that raises :class:`UnstableWarning` about ``what`` on each
    call, under the variable, and otherwise runs as it is. A call made from
    inside another labelled call does not warn again."""

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            depth = getattr(_inside, "depth", 0)
            if depth == 0:
                warn_unstable(what)
            _inside.depth = depth + 1
            try:
                return fn(*args, **kwargs)
            finally:
                _inside.depth = depth

        return wrapper

    return decorate


#: The attribute that marks a name kept only to refuse its old spelling: the
#: API snapshot and the reference leave such a name out, so both show the new
#: name only (``_renamed.is_renamed`` reads it).
RENAMED_MARK = "__rheon_renamed__"


class ReadinessWarning(RheonWarning, UserWarning):
    """A warning that a model is not ready, or may never become ready.

    It points once at a state every row already carries in
    ``withheld_reason``, ``settled_frac`` and ``support_coef``, so you can
    act before reading the output. Where the finding is intended, filter it
    out with ``warnings.simplefilter("ignore", rheon.ReadinessWarning)``.
    `Is it ready?
    <https://github.com/hgilde/polars-online/blob/main/docs/DIAGNOSTICS.md#is-it-ready>`_
    sets each finding beside the gates that withhold a prediction.

    The learning calls issue it: :meth:`ModelBank.fit`,
    :meth:`ModelBank.fit_predict`, :meth:`ModelBank.fit_predict_batches` and
    ``lf.rheon.fit_predict``. ``predict`` never does. Apart from the two
    findings about a horizon, it is issued once per spec and group. Its
    message begins with the spec, and the group where the spec has one, as
    ``spec "m" group "g":``. Under ``group_close="session"`` each session
    span is a stream of its own, so the warning is issued once per span,
    which the message names by its session value, as
    ``spec "m" group "g" session "a":`` (``session null`` for a null value).
    Two spans with the same finding then give two texts, and Python's
    default filter, which shows a text once per line of code, shows both.

    It reports five findings:

    - **A coefficient the ridge determined more than the data did**
      (``support_coef`` below 0.5), named, on a row the gates let through.
      The features do not determine such a coefficient: a duplicated or
      constant column, or a ridge as large as the feature's variance does
      this. While the columns stay collinear, the predictions do not depend
      on the split. The split among such
      columns is arbitrary, and it moves the moment the collinearity breaks.
    - **The ``max_error_inflation`` gate has not been met.** The gate has withheld every row
      for a further half-life, and at the row rate so far ``error_inflation``
      stays above ``max_error_inflation``, so predictions stay withheld
      unless the rows come faster. The message says how far, and gives a
      half-life above which the gate is met, or a looser ratio. Where the
      model has not fit at all, its ``weight_sum`` short of a row per
      coefficient, the message gives where ``weight_sum`` tops out instead.
      A ``kalman`` never issues it, since its gate reads each row's own
      value, which a row nearer the design's center reads lower.
    - **A** ``min_weight`` **has not been met.** The floor has withheld
      every row for a further half-life, and the weight a target reads tops
      out below its ``min_weight``. The message names that ceiling
      (``weight_sum_settled`` in :meth:`ModelBank.summary`), and the
      ``min_weight`` or ``half_life`` that meets it. A target that no row
      with a positive weight has given a value has no rate to project from,
      so its message says that instead. A target whose own weight is below
      ``weight_sum`` has not been learned from every row, and its weight
      falls on each row without it. Its message says so, with the ceiling
      the floor would have if every row carried the target, where that
      ceiling is below the floor.
    - **A diagnostic has no horizon.** A spec with a ``clock`` and an
      ``embargo`` but no ``horizon_rows`` has a diagnostic that reads the
      target's horizon (``emit_calibration``, ``emit_breaks``,
      ``emit_specification`` or ``emit_robust_se``). There ``embargo`` is in
      clock units, so the horizon is 0, and a target that looks ahead flags
      the test on most rows. The message names ``horizon_rows``, once per
      diagnostic, on the bank's first learning call.
    - **The Newey-West lags were capped at 64.** Without a clock,
      ``embargo`` is in rows and gives the horizon. A diagnostic that reads
      the horizon takes twice the horizon in Newey-West lags, and at most 64
      when they are read from ``embargo``. The
      message gives the lags asked for, and names ``horizon_rows`` (to take
      them all) and ``robust_se_lags``. It begins with the spec alone, and
      comes on the bank's first learning call.

    Offline, :func:`rheon.gram.lasso_path` issues it once per call
    for the penalties whose coordinate descent ran out of its ``max_iter``
    sweeps before it met its stopping rule, naming them. Each such
    penalty's row of the path is where its descent stopped. The ``lasso``
    model counts the same event in :meth:`ModelBank.solve_failures`, except
    on a Gram with too few rows for its features, its warm-up.

    .. rubric:: Examples

    This code reads ``df`` from `Example data
    <https://github.com/hgilde/polars-online/blob/main/docs/EXAMPLE-DATA.md>`_:

    .. code-block:: python

        import warnings

        # 400 rows a step apart, with a half-life of 20 steps: the weight tops out near 29.
        spec = rp.spec.ewridge(
            "ridge", targets=["y"], features=["x0", "x1"], clock="t", half_life=20.0,
            gap_cap=5.0, min_weight=50.0,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rp.ModelBank([spec]).fit_predict(df)
        for w in caught:
            print(type(w.message).__name__, str(w.message).split(" has not")[0])

    .. code-block:: text

        ReadinessWarning spec "ridge": min_weight = 50

    .. rubric:: See also

    - `Is it ready?
      <https://github.com/hgilde/polars-online/blob/main/docs/DIAGNOSTICS.md#is-it-ready>`_,
      the gates and what each finding means for the output.
    - :meth:`ModelBank.summary`, whose ``weight_sum_settled`` is the
      ceiling the ``min_weight`` finding names.
    - :class:`RheonWarning`, the base class every warning of this
      library shares.

    .. rubric:: Notes

    The first three findings judge a model only once its ``settled_frac``
    reaches 0.95, the model within 5% of its steady state. Before then, its
    early rows still move what the warning would say, and a warning cannot
    be taken back: the first rows' noisy feature variances, for example,
    can read a share of the data below 0.5 that later rows do not. Where
    nothing decays, ``settled_frac`` is null, so the model is never judged,
    and its ``support_coef``, ``withheld_reason`` and
    :meth:`ModelBank.summary` carry what it reads.

    The noise-gate and ``min_weight`` findings wait a half-life during which
    their gate withholds every row, on the clock the learned rows have
    covered. A row the gate lets through starts the wait again. The wait is
    in the state file: a bank saved and loaded inside it goes on waiting,
    and issues its warning on the row an unbroken stream would. Under
    ``embargo`` both findings read the ``settled_frac`` of the rows the
    model has learned from, on the clock its weight has covered. The row's
    own ``settled_frac`` counts the held rows' clock too.

    Both findings project a ceiling from the rows so far, and rows that come
    faster can beat it, so neither says the gate is unmet for good. At the
    rows' spacing ``d`` and weight so far, the ``min_weight`` ceiling is
    ``1 / (1 - 2 ** (-d / half_life))`` times that weight. With a clock,
    ``d`` is known only from the rows, so the ceiling cannot be read from the spec before
    the rows arrive. A warning on a row of
    weight 0, or on rows too uneven to project (one row far heavier than the
    rest, or a stream whose weight is still its first row's), gives no
    figure. A target the rows stopped carrying gets no figure where rows
    that each carried it would meet the floor, since its weight falls
    toward 0 once the rows stop carrying it.
    """


def _user_stacklevel() -> int:
    """The ``stacklevel`` that attributes a warning to the first frame outside
    this package, so it points at the caller's line and not at the library's.
    Counted from the function that calls this and then ``warnings.warn``."""
    here = os.path.dirname(os.path.abspath(__file__)) + os.sep
    frame: FrameType | None = sys._getframe(1)
    level = 1
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(here):
        frame = frame.f_back
        level += 1
    return level
