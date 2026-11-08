"""Generate `docs/OUTPUTS.md`: what each model writes into its output column.

The *field lists* and their *dtypes* come from `po.spec.output_index` on a
canonical spec per model, `MINIMAL` in `tests/test_model_registry.py`, so they
cannot drift from the code. The *meanings* and the rules for where a field is
null are written here, once per field stem, and reviewed like any other
prose: `ALSO_NULL` was measured on each model's plainest spec (review
2026-10-06, DA14). `FAMILIES` groups the models as the README's model table
and `docs/MODELS.md` do: the document's contents table and its section order follow it, and the
generator refuses to run while it leaves out a model of `MINIMAL` or names one
that is not there. `tests/test_outputs_doc.py` regenerates the document and
fails when it differs, so a new field cannot ship undocumented and a removed
one cannot linger.

Run: `uv run python scripts/outputs_doc.py > docs/OUTPUTS.md`
"""

from __future__ import annotations

import contextlib
import io
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import polars_online as po  # noqa: E402
from test_model_registry import MINIMAL, _build  # noqa: E402

#: The keywords `_build` gives every model before `MINIMAL`'s own.
BASE_KWARGS: dict[str, object] = {"targets": ["y"], "features": ["x0"], "half_life": 50.0}

#: The README's model table, family by family and in its order. The family's
#: name is also its heading in `docs/MODELS.md`, so the anchor is derived from it.
FAMILIES: dict[str, tuple[str, ...]] = {
    "Linear models": (
        "ewridge",
        "rls",
        "lasso",
        "kalman",
        "huber",
        "quantile",
        "sgd",
        "pa",
        "ftrl",
        "holt",
    ),
    "Moments and correlation": ("ew_cov", "marginal", "deco", "rcov", "audit"),
    "Clustering and classification": ("kmeans", "micro", "ew_class"),
    "Sequential tests and regimes": ("seqtest", "corrchange", "bocpd", "hmm"),
}

#: Where the four fields most models write are defined in full.
SHARED = "([shared field](#fields-most-models-write))"

#: One line per field *stem*. `<t>` is a target, `<f>` a feature, `<a>`/`<b>` a
#: pair of columns, `<j>` an index, `<label>` a class label.
MEANING: dict[str, str] = {
    "pred_<t>": "the prediction for `<t>`, computed from the state **before** this row",
    "pred_<f>": "the predictive mean for feature `<f>` under the fitted model",
    "resid_<t>": "`y - pred` for `<t>`",
    "weight_sum": (
        f"accumulated weight before the model learns this row, and before its own decay {SHARED}"
    ),
    "coef": f"the numbers behind the fit as one list {SHARED}",
    "settled_frac": (
        f"how far the model had come toward its steady state under decay before this row {SHARED}"
    ),
    "withheld_reason": f"why the row's predictions are null {SHARED}",
    "support_coef": (
        "on `coef`'s rows, each coefficient's data share `1 - ridge * (S^-1)_jj`, with `S` "
        "the system the solve inverts, laid out like `coef`"
    ),
    "penalty_selected_<t>": (
        "the penalty of the grid point in force for `<t>` before this row, the one with the "
        "lowest EW out-of-sample error"
    ),
    "mean_<f>": "EW mean of `<f>`",
    "var_<f>": "EW variance of `<f>`, corrected for the unequal weights as Polars' `ewm_var` "
    "is, unless `bias`",
    "std_<f>": "EW standard deviation of `<f>`, the square root of its EW variance",
    "cov_<a>_<b>": "EW covariance of the pair, corrected as `var_<f>` is",
    "corr_<a>_<b>": "EW correlation of the pair",
    "partial_corr_<a>_<b>": "the pair's correlation controlling for every other column",
    "mahal": "Mahalanobis distance of the row from the running mean, in standard deviations",
    "cluster": "the nearest cluster's label, read before the row is learned from",
    "dist": (
        "distance from the row to the center `cluster` was read from, in EW standard "
        "deviations under `standardize` and in the features' own units otherwise"
    ),
    "dist_second": "distance to the second-nearest center, so `dist_second - dist` is the margin",
    "micro_id": "id of the summary the row joins, or the one it opens when none can take it",
    "outlier": "true when no established summary takes the row",
    "n_clusters": "clusters currently linked",
    "n_micro": "live summaries",
    "class": "the most likely class label",
    "p_<label>": "posterior probability of class `<label>`",
    "log_e_pos_<t>": (
        "log of the e-process betting the sign of `<t>` is positive: 0 is no evidence, and "
        "`log(20)`, about 3.0, is evidence at level 0.05"
    ),
    "log_e_neg_<t>": "log of the e-process betting it is negative",
    "n_pos_<t>": "learned rows whose `<t>` was positive",
    "n_neg_<t>": "learned rows whose `<t>` was negative",
    "u": (
        "the equicorrelation this row alone implies"
        " (Engle and Kelly's Lemma 2.3, from the standardized row)"
    ),
    "rho": "the block's equicorrelation level, the smoothed value `u` is folded into",
    "stat": (
        "the test statistic, this row included, for the span, the pair of windows or the "
        "monitored row"
    ),
    "crit": "the critical value `stat` is compared against",
    "flag": "true on the row where `stat` crossed `crit`",
    "since_flag": "learned rows since the last flag",
    "since_change": (
        "on a flag, the rows since the change it dates, through the flag's row from the first "
        "changed one"
    ),
    "p_change": (
        "`P(run length <= 1)` once this row's evidence is in: the mass sitting on a change at "
        "or just before this row, the one `bocpd` field read after the row"
    ),
    "run_mode": (
        "most likely run length *before* this row, so the current run began `run_mode` rows "
        "before this one"
    ),
    "run_mean": "posterior mean run length",
    "loglik": "log predictive density of the row, under the model as it stood before the row",
    "filtered_state<j>": "posterior probability of state `<j>` before this row",
    "predicted_state<j>": "one-step-ahead probability of state `<j>`",
    "state": "the most likely state for this row: the one with the largest `predicted_state<j>`",
    "huber_scale_<t>": "the scale the row's cut for `<t>` is drawn in",
}

#: Where a field is null besides a skipped row, which nulls every field (see
#: *When a field is null* below): one line per field stem, measured on each
#: model's plainest spec with a null feature, a null target, a null weight and
#: a weight of 0 in its stream (review 2026-10-06, DA14).
WITHHELD = "while withheld"
UNTIL_SEEDED = (
    "while withheld, which includes the `warm_rows` learned rows (by default 1,000, or "
    "`n_components` where that is more) that seed the states"
)
EW_CLASS_NULL = (
    "while withheld, before any row has carried a class, and on a row whose class "
    "covariance cannot be factorized"
)
ALSO_NULL: dict[str, str] = {
    "pred_<t>": WITHHELD,
    "pred_<f>": WITHHELD,
    "resid_<t>": (
        "while withheld; where the target is null; and on every row of a target that is a "
        "formula target, whose value is not known at its row"
    ),
    "weight_sum": "never",
    "coef": (
        "on every row but those `coef_every` or `max_rows_between_coefs` fills (by default, each "
        "group's last row not skipped in each chunk), and before the model has anything "
        "to report"
    ),
    "settled_frac": "where nothing decays",
    "withheld_reason": "where nothing was withheld",
    "support_coef": "where `coef` is, and in the intercept's place, which is not a share",
    "penalty_selected_<t>": "never: it is written while the predictions are withheld too",
    "mean_<f>": WITHHELD,
    "var_<f>": WITHHELD,
    "std_<f>": WITHHELD,
    "cov_<a>_<b>": WITHHELD,
    "corr_<a>_<b>": "while withheld, and where either column's EW variance is 0",
    "partial_corr_<a>_<b>": WITHHELD,
    "mahal": WITHHELD,
    "dist": "where `cluster` is",
    "dist_second": "where `cluster` is",
    "micro_id": WITHHELD,
    "outlier": WITHHELD,
    "n_clusters": WITHHELD,
    "n_micro": WITHHELD,
    "class": EW_CLASS_NULL,
    "p_<label>": EW_CLASS_NULL,
    "log_e_pos_<t>": WITHHELD,
    "log_e_neg_<t>": WITHHELD,
    "n_pos_<t>": WITHHELD,
    "n_neg_<t>": WITHHELD,
    "u": (
        "while withheld, where fewer than two of the block's columns have a positive variance "
        "before the row, and where every standardized value is 0"
    ),
    "rho": "while withheld, and until the row after the first `u`",
    "stat": "while withheld, and on every row but those a test is due on",
    "crit": "where `stat` is",
    "flag": "where `stat` is",
    "since_flag": "where `stat` is",
    "since_change": (
        'on every row but a flag\'s; and on a `"sequential"` flag of the test between pairs '
        "(without `scalar`) that has fewer than two monitored rows before it, or where either "
        "column of the flagged pair is constant over those rows"
    ),
    "p_change": WITHHELD,
    "run_mode": WITHHELD,
    "run_mean": WITHHELD,
    "loglik": WITHHELD,
    "filtered_state<j>": UNTIL_SEEDED,
    "predicted_state<j>": UNTIL_SEEDED,
    "state": UNTIL_SEEDED,
    "huber_scale_<t>": "until the target has a scale above 0",
}

#: `bocpd`'s fields, null while its first rows set a prior left out (task 195).
BOCPD_STEMS = ("p_change", "run_mode", "run_mean", "pred_<f>", "loglik")
BOCPD_WARM = (
    "while withheld, which includes the first `warm_rows` learned rows (by default the feature "
    "count plus 2) where `mean_prior` or `covariance_prior` is left out, since they set the prior "
    "left out; and on a row whose predictive cannot be evaluated (counted by "
    "`ModelBank.solve_failures()`)"
)

#: A model whose rule for a stem differs from `ALSO_NULL`'s.
ALSO_NULL_BY_MODEL: dict[tuple[str, str], str] = {
    ("kmeans", "cluster"): (
        "while withheld, which includes the rows before seeding: seeding waits for `warm_rows` "
        "learned rows (at least `n_clusters`; by default 500, or `n_clusters` where that is "
        'more), and under `init="first"` past them until `n_clusters` distinct rows have '
        "come, holding at most `max(warm_rows, 1000)` rows"
    ),
    ("ewridge", "coef"): (
        "on every row but those `coef_every` or `max_rows_between_coefs` fills (by default, each "
        "group's last row not skipped in each chunk), on an `ill_conditioned` row, and "
        "before the model has anything to report"
    ),
    ("kmeans", "dist_second"): "where `cluster` is, and throughout when `n_clusters` is 1",
    ("kalman", "pred_<t>"): "while withheld, and on a target's first row, which has nothing to "
    "predict from",
    ("holt", "pred_<t>"): "while withheld, and on a target's first observation, which has "
    "nothing to predict from",
    ("deco", "loglik"): (
        "while withheld, where any column has no positive variance before the row, and where "
        "the levels hold no correlation matrix the density can be solved for"
    ),
    ("micro", "cluster"): "while withheld, and while no summary is established",
    ("hmm", "loglik"): UNTIL_SEEDED,
    **{("bocpd", stem): BOCPD_WARM for stem in BOCPD_STEMS},
}

#: The fields a model's own switch adds, shown under its table: the model, the
#: switch, and the spec keywords that turn it on.
SWITCHED: dict[str, tuple[str, dict[str, object]]] = {
    "huber": ("emit_huber_scale=True", {"emit_huber_scale": True}),
}

#: What a model writes when it writes nothing per row.
STATE_ONLY = {
    "audit": (
        "`audit` writes nothing per row but `weight_sum`, the rows before this one. Its product "
        "is the state, and `ModelBank.audit()` reads what it counted from it."
    ),
    "marginal": (
        "`marginal` writes nothing per row but `weight_sum`. Its product is the state, and "
        "`ModelBank.marginal()` reads the pairs from it."
    ),
    "rcov": (
        "`rcov` writes nothing per row but `weight_sum`. Its product is the closed block, in the "
        "row `ModelBank.closed_groups()` gives when a group closes (`group_close`). That row's "
        "`rcov_psd_repaired` is null where the repair could not run: on an estimate with an entry "
        "that is not finite, or where its eigendecomposition failed."
    ),
}

INTRO = """\
This reference page lists the fields each model writes into its output column,
one section per model, for a reader who wants to know what a field holds, its
type, and when it is null. Every spec adds one column to the output, named after
the spec, and its value in each row is a record of these fields. The tables are
generated from the code, and a test fails when they differ from it.
`po.spec.output_fields(spec)` answers the same question at runtime, for the
exact spec you built.
"""

#: Where the page comes from, for a contributor, as an HTML comment under the
#: kind line: a reader of the rendered page does not see it.
GENERATED = """\
<!--
scripts/outputs_doc.py writes this page from each model's plainest spec, the
one tests/test_model_registry.py builds. tests/test_outputs_doc.py regenerates
it and fails on any difference, so a new field cannot ship undocumented and a
removed one cannot linger. Do not edit this page by hand: change the
generator, then run `uv run python scripts/outputs_doc.py > docs/OUTPUTS.md`.
-->"""

#: The parts of a field's name, for the *Field names* table: the part, what it
#: stands for, and how it reads in this page's own tables.
NAME_PARTS: list[tuple[str, str, str]] = [
    ("`<t>`", "a target: its column, its `name`, or a formula target's alias", "`y`"),
    ("`<f>`", "a feature", "`x0`, `x1`"),
    ("`<a>`, `<b>`", "the two columns of a pair", "`x0`, `x1`"),
    ("`<j>`", "a state's index", "`0`, `1`"),
    ("`<label>`", "a class label", "`a`, `b`"),
    ("`__r<ridge>`", "one value of a `ridge` grid", ""),
    ("`__<set>`", "one of the `feature_sets`, with one ridge value", ""),
    ("`__<set>_r<ridge>`", "one of the `feature_sets`, with one value of a `ridge` grid", ""),
    ("`__p<penalty>`", "one value of `penalties`", "`__p0.1`, `__p0`"),
    (
        "`@h<half_life>`",
        "one half-life of a grid, at the end of every field's name: `weight_sum@h500`, or "
        "`weight_sum@h10m` for a duration",
        "",
    ),
]

#: The four fields most models write, in full: the field, what it holds, and
#: where else it is null. The dtype is read from `output_index`.
SHARED_FIELDS: list[tuple[str, str, str]] = [
    (
        "weight_sum",
        "the accumulated weight before the model learns this row, and before the row's own decay",
        "never",
    ),
    (
        "settled_frac",
        "how far the model had come toward its steady state under decay before this row: "
        "`1 - 2^(-T/half_life)`, with `T` the decay time seen so far, so 0.5 at one half-life "
        "and 0.75 at two. `min_settled_frac` gates on it",
        "where nothing decays",
    ),
    (
        "withheld_reason",
        "why the row's predictions are null: `below_min_settled_frac`, `below_min_weight`, "
        "`above_max_error_inflation`, `below_warm_rows`, `ill_conditioned` or `unstable_rate`. "
        "That order is their precedence, so the first that applies is the one named. "
        "`below_warm_rows` is the model's own warm-up, the rows `kmeans`, `hmm` and `bocpd` hold "
        "to seed themselves from, and, in a `kalman` fit standardized with no intercept, the "
        "rows before any of a target's coefficients is sized. `ill_conditioned` is an `ewridge` "
        "fit whose normal equations are too nearly singular for a solve to keep a digit. "
        "`unstable_rate` is an `sgd` target whose learning rate is past its stability bound: its "
        "recent steps' gains have a mean square above twice their mean, so each step grows its "
        "own row's residual and the fit diverges",
        "where nothing was withheld",
    ),
    (
        "coef",
        "the numbers behind the fit, as one flat list written after the row is learned: a "
        "regression's coefficients, or what a model that is not a regression keeps in their "
        "place, such as its centers or state means. Its builder's docstring lays the list "
        "out. A model that solves on a schedule (`solve_every`) shows its latest solve. The "
        "entries of a target or class with no fit yet are null: a target no solve has fit, a "
        "`holt` target not yet observed, an `ew_class` class not yet seen. `hmm` writes zeros, "
        "not nulls, on its seeding rows before the last. `ewridge` does not run a "
        "solve that comes due while its weight is too far below the floor for any row to "
        "read it before the next one, so there it shows the last fit solved. "
        "Under an `embargo`, learning the row waits for the delay, so `coef` is written "
        "after the rows this row releases, and is the fit the next row is predicted with "
        "only when the next row releases none",
        "on every row but those `coef_every` or `max_rows_between_coefs` fills, which with "
        "neither are each group's last row not skipped in each chunk; on an "
        "`ill_conditioned` row; and before the model has anything to report, such as a first "
        "solve",
    ),
]


def table(head: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> str:
    """A GitHub table, one row per tuple; an empty string is an empty cell."""

    def line(cells: tuple[str, ...]) -> str:
        return "|" + "|".join(f" {c} " if c else " " for c in cells) + "|"

    return "\n".join([line(head), "|" + "---|" * len(head), *map(line, rows)])


#: Where each optional output is shown, and the fields any spec can add.
OPTIONAL: list[tuple[str, str]] = [
    (
        "the diagnostics: `emit_sigma`, `emit_zscore`, `emit_selected`, "
        "`emit_averaged`, `emit_drift`, `emit_metrics`, `emit_autocorr`, `resid_quantiles`, "
        "`conformal`, `emit_calibration`, `emit_breaks`, `emit_specification`, "
        "`emit_tails`, `emit_influence` (`ewridge`'s, `rls`'s and `kalman`'s), "
        "`emit_feature_health` (not `holt`'s, which has no features)",
        "[Is the model working?](DIAGNOSTICS.md), for the models that "
        "predict a target. `emit_metrics`' `hit_rate_<t>` is null throughout on an `sgd` "
        'fit with `loss="poisson"`, whose rate and count have no sign to hit',
    ),
    (
        "`emit_error_inflation` and `emit_se_coef`, `ewridge`'s, `rls`'s and `kalman`'s",
        "[Warm-up gates](MODELS.md#warm-up-gates)",
    ),
    (
        "`emit_robust_se`, `ewridge`'s and `rls`'s",
        "[Can its t be trusted?](DIAGNOSTICS.md#can-its-t-be-trusted)",
    ),
    (
        "`emit_clocks`, every model's",
        "[Learn targets known late](how-to/learn-targets-known-late.md#check-that-it-worked),"
        " and the table of its two fields after this one",
    ),
    (
        "`emit_huber_scale`, `huber`'s: `huber_scale_<t>`, the scale each row's cut is drawn in",
        "[`huber`'s table](#huber), and its [entry](MODELS.md#huber--robust-regression)",
    ),
    (
        "`ew_cov`'s extra `stats`, its `lags` (`lag_corr_l<l>_<a>_<b>`), `pca` (`pc<j>_*`) and "
        "`mahal_quantiles` (`mahal_q<p>`)",
        "its own [section](MODELS.md#ew_cov--exponentially-weighted-moments) and its "
        "[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_cov)",
    ),
    (
        "`seqtest`'s `a` and `b`, which replace its four fields with `log_e_a_<t>`, "
        "`log_e_b_<t>`, `wins_a_<t>` and `wins_b_<t>`",
        "its own [section](MODELS.md#seqtest--a-sequential-test-of-a-sign-by-betting) and its "
        "[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.seqtest)",
    ),
    (
        "`deco`'s named blocks, which turn `u` and `rho` into one field per block and pair of "
        "blocks",
        "its [docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.deco)",
    ),
]
CLOCK_FIELDS: list[tuple[str, str, str]] = [
    (
        "scored_clock",
        "the clock the row was predicted at, in the clock column's own type; with no clock, the "
        "row's index in its group",
        "never",
    ),
    (
        "learned_clock",
        "the clock of the newest row the model had learned from at a positive weight when this "
        "row was predicted, in "
        "the same type: under an `embargo`, at least the delay behind `scored_clock`",
        "before the first row learned, and after a reset",
    ),
]

#: The header of every field table on the page.
FIELD_HEAD = ("field", "dtype", "what it holds", "also null")


def dtypes(spec: dict) -> dict[str, str]:
    """Each field's dtype, as the bank declares it to polars."""
    index = po.spec.output_index(spec)
    return dict(zip(index["field"], index["dtype"], strict=True))


def field_rows(rows: list[tuple[str, str, str]], spec: dict) -> list[tuple[str, ...]]:
    """`(field, holds, also null)` rows with the dtype `spec` declares for each."""
    declared = dtypes(spec)
    return [(f"`{f}`", f"`{declared[f]}`", holds, null) for f, holds, null in rows]


#: The specs the two tables of fields that are not one model's take their
#: dtypes from: the plainest `ewridge`, and the same with `emit_clocks`.
SHARED_SPEC = _build("ewridge")
CLOCK_SPEC = {**SHARED_SPEC, "emit_clocks": True}


#: Each dtype `output_index` names, and the polars type a field of it has.
DTYPES: list[tuple[str, str]] = [
    ("`f64`", "`Float64`"),
    ("`i64`", "`Int64`"),
    ("`bool`", "`Boolean`"),
    (
        "`list[f64]`",
        "`List(Float64)`: `coef` and the fields laid out like it. A list stores nothing for a "
        "null row, and `coef` is null on most rows. For an `(n, k)` matrix, `k` the height of "
        "`po.spec.coef_index(spec)`, keep the rows that carry it and fix their width. For the "
        "output `out` of a spec named `m`: "
        '`out.select(pl.col("m").struct.field("coef")).filter(pl.col("coef").is_not_null())'
        '.select(pl.col("coef").list.to_array(k)).to_series().to_numpy()`',
    ),
    (
        "`enum`",
        "an `Enum` of its values, in their order: `withheld_reason`'s six, an `ew_class` "
        "spec's `classes` for `class`, a grid's slots for `selected_<t>`",
    ),
    (
        "`clock`",
        "the clock column's own type, an integer clock in its own width; with no clock, "
        "`Int64`, the group's row index",
    ),
]


#: The recipe in *The fields of your own spec*: run here, so the page shows what it prints.
RECIPE = """\
import polars_online as po

grid = po.spec.ewridge(
    "ridge", targets=["y"], features=["x0", "x1"],
    half_life=[100.0, 500.0], ridge=[1e-6, 0.5],  # a grid: one fit per pair of values
)
print(po.spec.output_fields(grid))  # every field, by name
index = po.spec.output_index(grid)  # every field, with the values its name encodes
for row in index.filter(kind="pred").select("field", "target", "ridge", "half_life").iter_rows():
    print(*row)
"""


def printed(code: str) -> str:
    """What `code` prints, run in a namespace of its own."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "outputs_doc.RECIPE", "exec"), {"__name__": "__recipe__"})
    return out.getvalue().rstrip("\n")


def call(name: str) -> str:
    """The builder call `_build` makes for `name`, as a reader would type it."""
    kw = {**BASE_KWARGS, **MINIMAL[name]}
    args = ", ".join(f"{k}={v!r}" for k, v in kw.items() if v is not None)
    return f'po.spec.{name}("m", {args})'.replace("'", '"')


READING = f"""\
## The fields of your own spec

A spec of your own can carry a grid, several targets or another name, and
each changes its fields' names. Ask the spec itself:
`po.spec.output_fields` lists every field it writes, and
`po.spec.output_index` gives each field with the target, ridge and
half-life its name encodes, and its dtype. Here a spec with two half-lives
and two ridge values has four predictions of `y`, one per pair:

```python
{RECIPE}```

```text
{printed(RECIPE)}
```

## Field names

Each table lists the fields of the model's plainest spec, whose target is
`y` and whose features are `x0`, and `x1` where the model needs two. Your
own spec's fields carry your column names in their place.

**Every name follows one grammar, which is stable.** Read left to right, a
name is the kind (`pred`, `abs_resid`, `lag_corr`), then its qualifier, then
the target or the columns it is about (`_<t>`, `_<f>`, `_<a>_<b>`), then the
grid point after `__` and the instance after `@`. An instance is the part of
the model that fits one half-life of a grid. A qualifier is a number
glued to the letter that says what it is: `q` a quantile level, `l` a lag
(`abs_resid_q0.05_y`, `mahal_q0.99`, `lag_corr_l1_x0_x1`). The grid point
uses letters too, `r` a ridge and `p` a lasso penalty, and the instance `h`,
a half-life. A thing the spec has several of is numbered where it is named,
glued to its word: a principal component `pc0` (`pc0_var`), an `hmm` state
`state0` (`filtered_state0`), a `kmeans` center `cluster0`
(`coef_cluster0_x1`). A number in a name renders as a plain decimal in
`[1e-6, 1e7)`, as `0.000001`, and in compact scientific notation outside
it, as `1e-300`. [Find an output field by what it
holds](how-to/find-an-output-field.md) builds a grid and reads its fields.

{table(("in a name", "stands for", "in this page's tables"), NAME_PARTS)}

EW, in a *what it holds* cell, is short for exponentially weighted.

## Field types

A field's dtype is the type the bank declares to Polars before it reads a
row, as `po.spec.output_index(spec)` lists it:

{table(("dtype", "the Polars type"), DTYPES)}

**Every integer is an `Int64`**, in a field and in every frame the bank and
`po.spec` return: a count, a state, a cluster label, a position, a row's
index. It is Polars' own default integer, so a join, a concatenation or a
comparison between two frames never meets two widths. Only the clock values
differ, in a field or in a frame such as `groups()`' `clock_last`: an
integer clock keeps its own width.

## When a field is null

**Every field is null on a skipped row**: a row whose features or weight
hold a null, or a value that counts as one (NaN, ±inf, or a magnitude above
`1e100`). The model learns nothing from such a row. `audit` skips no row:
it reads every row and counts what it cannot use. A row of weight 0 is
not skipped: it is predicted, and the model learns nothing from it.

**Most of a model's own fields are also null on a withheld row**: a row
whose outputs a gate held back, or whose fit cannot be trusted there.
`withheld_reason` names which ([Warm-up gates](MODELS.md#warm-up-gates)). Each
table's last column, *also null*, says where a field is null besides a
skipped row, and *withheld* there means such a row.

## Fields most models write

Four fields appear in nearly every model's table, and are defined here once:

{table(FIELD_HEAD, field_rows(SHARED_FIELDS, SHARED_SPEC))}

## What is not listed

The optional outputs are left out of the tables: the `emit_*` switches (all
but `emit_clocks` and `huber`'s `emit_huber_scale`, whose fields are listed),
`conformal`, `resid_quantiles`, and the options that add or rename a model's
fields, such as `ew_cov`'s `stats`. Each optional output is shown where
the docs set it:

{table(("the switch", "where the docs show it"), OPTIONAL)}

`emit_clocks` is the one switch every model takes, and it adds two fields:

{table(FIELD_HEAD, field_rows(CLOCK_FIELDS, CLOCK_SPEC))}
"""


def stem(field: str, targets: tuple[str, ...], features: tuple[str, ...]) -> str:
    """The `MEANING` key a realized field name belongs to."""
    base, _, _grid = field.partition("__")
    # Features first: the unsupervised models name a *feature* as their
    # nominal target, so `pred_x0` there is a feature's predictive mean and
    # not a regression's prediction.
    for f in features:
        if base == f"pred_{f}":
            return "pred_<f>"
        for pre in ("mean", "var", "std"):
            if base == f"{pre}_{f}":
                return f"{pre}_<f>"
    for t in targets:
        for pre in (
            "pred",
            "resid",
            "huber_scale",
            "penalty_selected",
            "log_e_pos",
            "log_e_neg",
            "n_pos",
            "n_neg",
        ):
            if base == f"{pre}_{t}":
                return f"{pre}_<t>"
    for pre in ("corr", "cov", "partial_corr"):
        if base.startswith(pre + "_"):
            return f"{pre}_<a>_<b>"
    for pre in ("filtered_state", "predicted_state"):
        if base.startswith(pre) and base[len(pre) :].isdigit():
            return f"{pre}<j>"
    if base.startswith("p_") and base != "p_change":
        return "p_<label>"
    return base


def check_families() -> None:
    """`FAMILIES` must place every model of `MINIMAL` exactly once."""
    listed = [name for names in FAMILIES.values() for name in names]
    missing = sorted(set(MINIMAL) - set(listed))
    unknown = sorted(set(listed) - set(MINIMAL))
    twice = sorted({name for name in listed if listed.count(name) > 1})
    if missing or unknown or twice:
        raise SystemExit(
            "outputs_doc.py: FAMILIES must name every model of MINIMAL exactly once; "
            f"missing {missing}, not a model {unknown}, named twice {twice}"
        )


def main() -> None:
    check_families()
    print("<!-- kind: reference -->")
    print(GENERATED)
    print("# What each model writes\n")
    print(INTRO)
    print("| family | models |")
    print("|---|---|")
    for family, names in FAMILIES.items():
        anchor = family.lower().replace(" ", "-")
        links = " · ".join(f"[`{name}`](#{name})" for name in names)
        print(f"| [{family}](MODELS.md#{anchor}) | {links} |")
    print()
    print(READING)
    for names in FAMILIES.values():
        for name in names:
            spec = _build(name)
            declared = dtypes(spec)
            targets = tuple(spec.get("targets") or ())
            features = tuple(spec.get("features") or ())
            print(f"\n## `{name}`\n")
            if name in STATE_ONLY:
                print(f"{STATE_ONLY[name]}\n")
            print(f"The fields of `{call(name)}`:\n")
            rows = []
            for f, dtype in declared.items():
                s = stem(f, targets, features)
                null = ALSO_NULL_BY_MODEL.get((name, s), ALSO_NULL.get(s, "**undocumented**"))
                rows.append((f"`{f}`", f"`{dtype}`", MEANING.get(s, "**undocumented**"), null))
            print(table(FIELD_HEAD, rows))
            if name in SWITCHED:
                switch, extra = SWITCHED[name]
                kw = {**BASE_KWARGS, **MINIMAL[name], **extra}
                more = dtypes(
                    getattr(po.spec, name)("m", **{k: v for k, v in kw.items() if v is not None})
                )
                rows = []
                for f, dtype in more.items():
                    if f in declared:
                        continue
                    s = stem(f, targets, features)
                    null = ALSO_NULL.get(s, "**undocumented**")
                    rows.append((f"`{f}`", f"`{dtype}`", MEANING.get(s, "**undocumented**"), null))
                print(f"\nUnder `{switch}`, it also writes:\n")
                print(table(FIELD_HEAD, rows))


if __name__ == "__main__":
    main()
