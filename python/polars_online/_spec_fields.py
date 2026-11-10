"""The spec helpers that name a bank's output fields: :func:`output_fields`,
:func:`output_index`, :func:`coef_fields` and :func:`coef_index`, exported
as ``polars_online.spec``'s. They read a spec as the bank does, through the
extension, and live apart from the builders in ``_spec.py``.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from polars_online import _spec
from polars_online._polars_online import spec_coef_fields, spec_output_fields, spec_output_index
from polars_online._spec import _json


def output_fields(spec: dict[str, Any]) -> list[str]:
    """List the names of the fields a spec writes, in order.

    Use it to see a spec's fields before any row is fed: the fields are fixed
    by the spec. For a whole bank, read
    :attr:`polars_online.ModelBank.output_fields`, and for what each field
    is, with the values its name encodes, use :func:`output_index`.

    .. rubric:: Parameters

    ``spec``
        A spec, as a builder of :mod:`polars_online.spec` returns it.

    .. rubric:: Returns

    The field names, in the order the output column holds them.

    .. rubric:: Raises

    ``ValueError``
        ``spec`` is not a valid spec, with the builders' message. A value
        that is not a spec at all is told what it lacks, such as
        ``invalid spec: missing field `name```.

    .. rubric:: Examples

    .. code-block:: python

        spec = po.spec.ewridge(
            "ridge", targets=["y"], features=["x0", "x1"], half_life=100.0, ridge=[1e-6, 0.1]
        )
        fields = po.spec.output_fields(spec)
        print(fields[:4])                         # each ridge value's prediction and residual
        print(fields[4:])                         # the fields the ridge values share

    .. code-block:: text

        ['pred_y__r0.000001', 'resid_y__r0.000001', 'pred_y__r0.1', 'resid_y__r0.1']
        ['weight_sum', 'settled_frac', 'withheld_reason', 'coef', 'support_coef']

    .. rubric:: See also

    - :func:`output_index`, each field with the values its name encodes.
    - `OUTPUTS.md
      <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md>`_,
      every model's fields.
    """
    return spec_output_fields(_json(spec))


def output_index(spec: dict[str, Any]) -> pl.DataFrame:
    """List every field a spec writes, with the values its name encodes.

    Use it to reach a field without building its name by hand, by filtering
    on what the field is about, such as its target, its ridge value and its
    half-life. For the names alone, use :func:`output_fields`, and for each
    coefficient's place in a ``coef`` list, :func:`coef_fields`.

    .. rubric:: Parameters

    ``spec``
        A spec, as a builder of :mod:`polars_online.spec` returns it.

    .. rubric:: Returns

    A frame with one row per field, in the output column's order. No two
    rows hold the same values apart from ``field``, so a filter on the other
    columns reaches each field alone. Its columns:

    ``field``
        The name, as it appears in the output column.
    ``kind``
        What it is: ``pred``, ``resid``, ``sigma``, ``weight_sum``, ``coef``,
        ``penalty_selected``, ``selected``, a statistic's stem, and so on.
    ``target``
        The target the field is about, or null.
    ``instance``
        The instance the field belongs to, by its suffix, as
        :meth:`~polars_online.ModelBank.coef`, :meth:`~polars_online.ModelBank.gram`
        and :meth:`~polars_online.ModelBank.closed_groups` name it: ``"@h500"``,
        or ``""`` for a single instance. Null on a field that chooses across
        the instances (``emit_selected``, ``emit_averaged``) and on the clocks,
        which are the spec's own.
    ``half_life``, ``lam``
        The decay of the instance the field belongs to, filled for a single
        instance as for a grid. ``half_life`` is the instance's half-life in
        clock units, in seconds under a duration (``300.0`` for ``"5m"``),
        ``inf`` for an infinite one, and null under ``lam``. ``lam`` is the
        spec's ``lam``, and null under a half-life.
    ``ridge``, ``feature_set``, ``penalty``
        The grid point: the ridge value, the feature set's name and the
        lasso penalty, each null where the spec has no such grid.
    ``quantile``
        The level of a ``resid_quantiles`` or ``mahal_quantiles`` field.
    ``lag``
        The lag, in learned rows, of an ``ew_cov`` ``lag_corr`` field.
    ``component``
        The principal component of an ``ew_cov`` ``pc<j>_*`` field.
    ``state``
        The hidden state an ``hmm`` posterior is about (``filtered_state<s>``,
        ``predicted_state<s>``).
    ``class``
        The class an ``ew_class`` posterior is about (``p_<class>``).
    ``columns``
        The columns the field is computed from:

        - one for an ``ew_cov`` mean, variance or standard deviation, for a
          feature-health field, and for ``bocpd``'s predictive mean of a
          column (``pred_<column>``);
        - the pair for a correlation or a covariance, ordered for ``lag_corr``;
        - every feature for ``mahal``, the principal components and the other
          fields of ``kmeans``, ``micro``, ``ew_class``, ``hmm``, ``bocpd`` and
          ``corrchange``;
        - a block or a pair of blocks for ``deco``.

        Null where no column set describes the field, as for a prediction.
    ``dtype``
        The type the bank declares to Polars before the first row is read:
        ``f64``, ``bool``, ``list[f64]``, ``enum`` (an ``Enum`` of
        its values in their order: ``withheld_reason``'s, an ``ew_class``
        spec's ``classes`` for ``class``, a grid's slots for
        ``selected_<t>``), ``i64`` (every integer: a count, a state, a row
        index, an id or a cluster label), or ``clock`` (``scored_clock`` and
        ``learned_clock``, in the clock column's own type).

    .. rubric:: Raises

    ``ValueError``
        ``spec`` is not a valid spec, as for :func:`output_fields`.

    .. rubric:: Examples

    This code reads ``df`` from `Example data
    <https://github.com/hgilde/polars-online/blob/main/docs/EXAMPLE-DATA.md>`_:

    .. code-block:: python

        grid = po.spec.ewridge(
            "m", targets=["y"], features=["x0", "x1"], half_life=[100.0, 500.0], ridge=[1e-6, 0.5]
        )
        out = po.ModelBank([grid]).fit_predict(df)    # a half-life grid by a ridge grid
        idx = po.spec.output_index(grid)
        name = idx.filter(
            (pl.col("kind") == "pred")
            & (pl.col("target") == "y")
            & (pl.col("ridge") == 0.5)
            & (pl.col("half_life") == 500.0)
        )["field"].item()
        print(name, out["m"].struct.field(name).count())

    .. code-block:: text

        pred_y__r0.5@h500 396

    .. rubric:: See also

    - :func:`output_fields`, the names alone.
    - :func:`coef_fields`, each coefficient's place in ``coef``.
    - `OUTPUTS.md
      <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md>`_,
      every model's fields.

    .. rubric:: Notes

    **Every name follows one grammar, which is stable.** Read left to right, a
    name is the kind (``pred``, ``abs_resid``, ``lag_corr``), then its
    qualifier, then the target or the columns it is about, then the grid point
    after ``__`` and the instance after ``@``. A qualifier is a number glued to
    the letter that says what it is: ``q`` a quantile level, ``l`` a lag
    (``abs_resid_q0.05_y``, ``mahal_q0.99``, ``lag_corr_l1_x0_x1``). The grid
    point uses letters too, ``r`` a ridge and ``p`` a lasso penalty, and the
    instance ``h``, a half-life. A thing the spec has several of is numbered
    where it is named, glued to its word: ``pc0_var``, ``filtered_state0``,
    ``pc0_loading_x1``.

    The table is made by the Rust code that writes the names, so it cannot
    drift from them.
    """
    return spec_output_index(_json(spec))


def coef_fields(spec: dict[str, Any]) -> pl.DataFrame:
    """List every coefficient a spec reports, with its place in ``coef`` and its unnested column.

    Use it to reach one coefficient in the nested output without writing
    its field's name or its position by hand, or to read the column names
    :meth:`~polars_online._frame.LazyFrameOnlineNamespace.unnest` gives.
    For one ``coef`` list's layout alone, use :func:`coef_index`.

    .. rubric:: Parameters

    ``spec``
        A spec, as a builder of :mod:`polars_online.spec` returns it.

    .. rubric:: Returns

    A frame with one row per instance, target, grid point and term. It is
    empty, with the same columns, for the kinds that report no
    coefficients (``ew_cov``, ``seqtest``, ``marginal``, ``rcov``,
    ``corrchange``, ``bocpd`` and ``audit``), and for ``micro``, whose
    ``coef`` has no fixed layout. Its columns:

    ``field``
        The ``coef`` list the coefficient sits in: ``coef``, or
        ``coef@h500`` for each half-life's instance.
    ``position``
        Its index in that list.
    ``name``
        The column
        :meth:`~polars_online._frame.LazyFrameOnlineNamespace.unnest` gives
        it: ``coef_{target}_{term}{combo}{instance}``, so
        ``coef_y_x1__r0.5@h500`` sits beside ``pred_y__r0.5@h500``. The
        ``coef_`` prefix and the target are there because a bare ``x1``
        would collide with the feature column of that name in the same
        frame. :func:`deco`'s are ``coef_{entry}{instance}``, its entry
        ``rho`` or a block or pair of blocks (``coef_tech``,
        ``coef_tech_banks``).
    ``target``, ``instance``, ``half_life``, ``lam``, ``ridge``, ``feature_set``, ``penalty``
        As :func:`output_index` reports them. ``target`` is
        :func:`ew_class`'s label column, a :func:`deco` entry, and null for
        a :func:`kmeans` center and an :func:`hmm` state, which belong to no
        target.
    ``cluster``, ``state``, ``class``
        The :func:`kmeans` center, the :func:`hmm` state or the
        :func:`ew_class` class whose coordinates the entry holds, and null
        for every other kind.
    ``term``
        ``"intercept"``, a feature name, ``"level"`` or ``"trend"`` for
        :func:`holt`, or ``"rho"`` for :func:`deco`.

    .. rubric:: Raises

    ``ValueError``
        ``spec`` is not a valid spec, as for :func:`output_fields`.

    .. rubric:: Examples

    This code reads ``df`` from `Example data
    <https://github.com/hgilde/polars-online/blob/main/docs/EXAMPLE-DATA.md>`_:

    .. code-block:: python

        grid = po.spec.ewridge(
            "m", targets=["y"], features=["x0", "x1"], half_life=[100.0, 500.0], ridge=[1e-6, 0.5]
        )
        out = po.ModelBank([grid]).fit_predict(df)    # a half-life grid by a ridge grid
        cf = po.spec.coef_fields(grid)
        row = cf.filter(
            (pl.col("target") == "y")
            & (pl.col("term") == "x1")
            & (pl.col("ridge") == 0.5)
            & (pl.col("half_life") == 500.0)
        ).row(0, named=True)
        slope = out["m"].struct.field(row["field"]).list.get(row["position"])
        print(row["field"], row["position"], row["name"])

    .. code-block:: text

        coef@h500 5 coef_y_x1__r0.5@h500

    .. rubric:: See also

    - :func:`coef_index`, one ``coef`` list's layout.
    - :func:`output_index`, every field with the values its name encodes.
    - :meth:`~polars_online._frame.LazyFrameOnlineNamespace.unnest`, which
      writes these columns.

    .. rubric:: Notes

    The table is made by the Rust code that names the fields, from the
    order the models lay the list out in. That order is the intercept first
    when the spec has one, then every feature, each target and grid point
    in turn, in the order the ``pred`` fields are declared. A feature
    outside a feature set reads zero.
    """
    return spec_coef_fields(_json(spec))


#: :func:`coef_index`'s columns, of a :func:`coef_fields` frame.
_INDEX_COLUMNS = (
    "position",
    "target",
    "ridge",
    "feature_set",
    "penalty",
    "cluster",
    "state",
    "class",
    "term",
)


def _index_columns(fields: pl.DataFrame) -> pl.DataFrame:
    """:func:`coef_index`'s columns of a :func:`coef_fields` frame: the one
    place they are chosen, so a bank with no coefficients lays out the empty
    frame :meth:`ModelBank.coef` gives with them too (review round 4, SF3)."""
    return fields.select(_INDEX_COLUMNS)


def _coef_index_schema() -> pl.Schema:
    """The schema of every :func:`coef_index` frame, whatever the spec: the
    empty layout of a spec with no coefficients."""
    return _index_columns(
        coef_fields(_spec.ew_cov("_", features=["_"], stats=["mean"], half_life=1.0))
    ).schema


def coef_index(spec: dict[str, Any]) -> pl.DataFrame:
    """List the layout of a spec's ``coef`` lists, one row per position.

    Use it to find which position of a ``coef`` list holds a coefficient.
    Every instance's list has this layout. :func:`coef_fields` is the same
    table for each instance, with the field each list sits in and the
    column each entry unnests to.

    .. rubric:: Parameters

    ``spec``
        A spec, as a builder of :mod:`polars_online.spec` returns it.

    .. rubric:: Returns

    A frame with one row per position. ``coef`` is flat: for each target
    and grid point in turn, its terms in order. The frame maps ``position``
    to ``target``, the grid point's ``ridge``, ``feature_set`` and
    ``penalty``, and ``term``: ``"intercept"``, a feature name, or
    ``"level"`` or ``"trend"`` for :func:`holt`.

    - For :func:`kmeans` the entries are the centers, so ``cluster`` reads
      ``0``, ``1``, and so on, and ``term`` is the feature whose coordinate
      the position holds.
    - For :func:`hmm` an entry is a state, in ``state``, and for
      :func:`ew_class` a class, in ``class``.
    - For :func:`deco` an entry is ``rho``, or a block or a pair of blocks,
      in ``target``, and ``term`` reads ``"rho"``.

    It is empty, with the same columns, for a spec with no fixed layout to
    index, as :func:`coef_fields` is:

    - ``ew_cov``, which has no coefficients;
    - ``seqtest``, which reports evidence;
    - ``micro``, whose ``coef`` has as many entries as there are established
      summaries;
    - ``marginal``, ``rcov``, ``corrchange``, ``bocpd`` and ``audit``, which
      report none.

    .. rubric:: Raises

    ``ValueError``
        ``spec`` is not a valid spec, as for :func:`output_fields`.

    .. rubric:: Examples

    This code reads ``df`` from `Example data
    <https://github.com/hgilde/polars-online/blob/main/docs/EXAMPLE-DATA.md>`_:

    .. code-block:: python

        grid = po.spec.ewridge(
            "m", targets=["y"], features=["x0", "x1"], half_life=[100.0, 500.0], ridge=[1e-6, 0.5]
        )
        out = po.ModelBank([grid]).fit_predict(df)    # a half-life grid by a ridge grid
        ci = po.spec.coef_index(grid)
        pos = ci.filter(
            (pl.col("target") == "y") & (pl.col("ridge") == 0.5) & (pl.col("term") == "x1")
        )["position"].item()
        slope = out["m"].struct.field("coef@h100").list.get(pos)
        print(ci.height, pos)

    .. code-block:: text

        6 5

    .. rubric:: See also

    - :func:`coef_fields`, the same for each instance, with each entry's
      column name.
    - :func:`output_index`, every field with the values its name encodes.
    """
    cf = coef_fields(spec)
    if cf.is_empty():
        return _index_columns(cf)
    return _index_columns(cf.filter(pl.col("field") == cf["field"][0]))
