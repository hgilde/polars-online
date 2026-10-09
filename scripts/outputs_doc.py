"""Generate `docs/OUTPUTS.md`: what each model writes into its output column.

The *field lists* and their *dtypes* come from `po.spec.output_index` on a
canonical spec per model, `MINIMAL` in `tests/test_model_registry.py`, so they
cannot drift from the code. The *meanings* and the rules for where a field is
null are written here, once per field stem, and reviewed like any other
prose: `ALSO_NULL` was measured on each model's plainest spec (review
2026-10-06, DA14). `FAMILIES` groups the models as the README's model table
does: the document's contents table and its section order follow it, and the
generator refuses to run while it leaves out a model of `MINIMAL` or names one
that is not there. `tests/test_outputs_doc.py` regenerates the document and
fails when it differs, so a new field cannot ship undocumented and a removed
one cannot linger.

Run: `uv run python scripts/outputs_doc.py > docs/OUTPUTS.md`
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import polars_online as po  # noqa: E402
from test_model_registry import MINIMAL, _build  # noqa: E402

#: The README's model table, family by family and in its order. The family's
#: name is also its heading there, so the anchor is derived from it.
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
    "weight_sum": f"accumulated weight before this row's update and before its own decay {SHARED}",
    "coef": f"the numbers behind the fit as one list {SHARED}",
    "settled_frac": f"how far the decay window had filled before this row {SHARED}",
    "withheld_reason": f"why the row's predictions are null {SHARED}",
    "support_coef": (
        "on `coef`'s rows, each coefficient's data share `1 - ridge * (S^-1)_jj`, laid "
        "out like `coef`"
    ),
    "penalty_selected_<t>": "the path point in force for `<t>`, by lowest EW out-of-sample error",
    "mean_<f>": "EW mean of `<f>`",
    "var_<f>": "EW variance of `<f>`, in the population form: no `ddof` correction",
    "std_<f>": "EW standard deviation of `<f>`, the square root of the population variance",
    "cov_<a>_<b>": "EW covariance of the pair, in the population form: no `ddof` correction",
    "corr_<a>_<b>": "EW correlation of the pair",
    "partial_corr_<a>_<b>": "the pair's correlation controlling for every other column",
    "mahal": "Mahalanobis distance of the row from the running mean, in standard deviations",
    "cluster": "the nearest cluster's label, read before the row is learned from",
    "dist": "distance from the row to the centre `cluster` was read from",
    "dist_second": "distance to the second-nearest centre, so `dist_second - dist` is the margin",
    "micro_id": "id of the micro-cluster the row joins, or opens when none can take it",
    "outlier": "true when no established micro-cluster takes the row",
    "n_clusters": "macro-clusters currently linked",
    "n_micro": "live micro-clusters",
    "class": "the most likely class label",
    "p_<label>": "posterior probability of class `<label>`",
    "log_e_pos_<t>": (
        "log of the e-process betting the sign of `<t>` is positive: 0 is no evidence, and "
        "`log(20)`, 3.0, is evidence at level 0.05"
    ),
    "log_e_neg_<t>": "log of the e-process betting it is negative",
    "n_pos_<t>": "learned rows whose `<t>` was positive",
    "n_neg_<t>": "learned rows whose `<t>` was negative",
    "u": "the equicorrelation this row alone implies (Lemma 2.3, from the standardized row)",
    "rho": "the block's equicorrelation level, the smoothed value `u` is folded into",
    "stat": "the test statistic for the span, the pair of windows or the monitored row",
    "crit": "the critical value `stat` is compared against",
    "flag": "true on the row where `stat` crossed `crit`",
    "since_flag": "learned rows since the last flag",
    "since_change": (
        "on a flag, the rows since the change it dates, through the flag's row from the first "
        "changed one"
    ),
    "p_change": "`P(run length <= 1)`: the mass sitting on a change at or just before this row",
    "run_mode": "most likely run length *before* this row, so `t - run_mode` dates the regime",
    "run_mean": "posterior mean run length",
    "loglik": "log predictive density of the row, under the model as it stood before the row",
    "filtered_<j>": "posterior probability of state `<j>` before this row",
    "predicted_<j>": "one-step-ahead probability of state `<j>`",
    "state": "the most likely state for this row: the one with the largest `predicted_<j>`",
}

#: Where a field is null besides a skipped row, which nulls every field (see
#: *When a field is null* below): one line per field stem, measured on each
#: model's plainest spec with a null feature, a null target, a null weight and
#: a weight of 0 in its stream (review 2026-10-06, DA14).
WITHHELD = "while withheld"
UNTIL_SEEDED = (
    "while withheld, and until `warm_rows` learned rows (default 50) have seeded the states"
)
ALSO_NULL: dict[str, str] = {
    "pred_<t>": WITHHELD,
    "pred_<f>": WITHHELD,
    "resid_<t>": (
        "while withheld; where the target is null; and on every row of a target that is a "
        "window expression, whose value is not known at its row"
    ),
    "weight_sum": "never",
    "coef": (
        "on the rows `coef_every` or `max_rows_between_coefs` does not fill, and before the "
        "model has anything to report"
    ),
    "settled_frac": "where nothing decays",
    "withheld_reason": "where nothing was withheld",
    "support_coef": "where `coef` is, and in the intercept's place, which is not a share",
    "penalty_selected_<t>": "never: it is written while the predictions are withheld too",
    "mean_<f>": WITHHELD,
    "var_<f>": WITHHELD,
    "std_<f>": WITHHELD,
    "cov_<a>_<b>": WITHHELD,
    "corr_<a>_<b>": WITHHELD,
    "partial_corr_<a>_<b>": WITHHELD,
    "mahal": WITHHELD,
    "dist": "where `cluster` is",
    "dist_second": "where `cluster` is",
    "micro_id": WITHHELD,
    "outlier": WITHHELD,
    "n_clusters": WITHHELD,
    "n_micro": WITHHELD,
    "class": WITHHELD,
    "p_<label>": WITHHELD,
    "log_e_pos_<t>": WITHHELD,
    "log_e_neg_<t>": WITHHELD,
    "n_pos_<t>": WITHHELD,
    "n_neg_<t>": WITHHELD,
    "u": WITHHELD,
    "rho": WITHHELD,
    "stat": "while withheld, and on every row but those a test is due on",
    "crit": "where `stat` is",
    "flag": "where `stat` is",
    "since_flag": "where `stat` is",
    "since_change": "on every row but a flag's",
    "p_change": WITHHELD,
    "run_mode": WITHHELD,
    "run_mean": WITHHELD,
    "loglik": WITHHELD,
    "filtered_<j>": UNTIL_SEEDED,
    "predicted_<j>": UNTIL_SEEDED,
    "state": UNTIL_SEEDED,
}

#: `bocpd`'s fields, null while its first rows set a prior left out (task 195).
BOCPD_STEMS = ("p_change", "run_mode", "run_mean", "pred_<f>", "loglik")
BOCPD_WARM = (
    "while withheld, and on the first `warm_rows` learned rows (by default the feature count "
    "plus 2) where `prior_mean` or `prior_scale` is left out, which set it"
)

#: A model whose rule for a stem differs from `ALSO_NULL`'s.
ALSO_NULL_BY_MODEL: dict[tuple[str, str], str] = {
    ("kmeans", "cluster"): (
        "while withheld, and before seeding, which waits for `warm_rows` learned rows (at "
        "least `k`; by default 500, or `k` where that is more)"
    ),
    ("micro", "cluster"): "while withheld, and while no micro-cluster is established",
    ("hmm", "loglik"): UNTIL_SEEDED,
    **{("bocpd", stem): BOCPD_WARM for stem in BOCPD_STEMS},
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
        "`rcov_psd_repaired` is null where the repair could not run, on an estimate with an entry "
        "that is not finite."
    ),
}

INTRO = """\
Every spec adds one column to the output, named after the spec. Its value in
each row is a record of named fields, and this page lists the fields each
model writes, one section per model. `po.spec.output_fields(spec)` answers
the same question at runtime, for the exact spec you built.
"""

#: The parts of a field's name, for the *Field names* table: the part, what it
#: stands for, and how it reads in this page's own tables.
NAME_PARTS: list[tuple[str, str, str]] = [
    ("`<t>`", "a target: its column, its `name`, or a window target's alias", "`y`"),
    ("`<f>`", "a feature", "`x0`, `x1`"),
    ("`<a>`, `<b>`", "the two columns of a pair", "`x0`, `x1`"),
    ("`<j>`", "a state's index", "`0`, `1`"),
    ("`<label>`", "a class label", "`a`, `b`"),
    ("`__r<ridge>`", "one value of a `ridge` grid", ""),
    ("`__<set>`", "one of the `feature_sets`, with one ridge value", ""),
    ("`__<set>_r<ridge>`", "one of the `feature_sets`, with one value of a `ridge` grid", ""),
    ("`__l<lambda>`", "one value of `lasso_path`", "`__l0.1`, `__l0`"),
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
        "the accumulated weight before this row's update and before its own decay",
        "never",
    ),
    (
        "settled_frac",
        "how far the decay window had filled toward steady state before this row: "
        "`1 - 2^(-T/half_life)`, with `T` the decay time seen so far, so 0.5 at one half-life "
        "and 0.75 at two. `min_settled_frac` gates on it",
        "where nothing decays",
    ),
    (
        "withheld_reason",
        "why the row's predictions are null: `below_min_settled_frac`, `below_min_weight` "
        "or `above_max_error_inflation`. That order is their precedence, so the first that "
        "applies is the one named",
        "where nothing was withheld",
    ),
    (
        "coef",
        "the numbers behind the fit, as one flat list written after the row's update: a "
        "regression's coefficients, or what a model that is not a regression keeps in their "
        "place, such as its centres or state means. Its builder's docstring lays the list "
        "out. A model that solves on a schedule (`solve_every`) shows its latest solve, and "
        "the entries of a target no solve has fit yet are null. "
        "Under an `embargo` the row's own update waits for the delay, so `coef` is written "
        "after the rows this row releases, and is the fit the next row is predicted with "
        "only when the next row releases none",
        "on every row but those `coef_every` or `max_rows_between_coefs` fills, which with "
        "neither are each group's last accepted row in each chunk; and before the model has "
        "anything "
        "to report, such as a first solve",
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
        "the residual diagnostics: `emit_sigma`, `emit_zscore`, `emit_selected`, "
        "`emit_averaged`, `emit_drift`, `emit_metrics`, `emit_autocorr`, `resid_quantiles`, "
        "`conformal`, `emit_calibration`",
        "[Per-row diagnostics](../README.md#per-row-diagnostics), for the models that "
        "predict a target. `emit_metrics`' `hit_rate_<t>` is null throughout on an `sgd` "
        'fit with `loss="poisson"`, whose rate and count have no sign to hit',
    ),
    (
        "`emit_error_inflation` and `emit_se_coef`, `ewridge`'s, `rls`'s and `kalman`'s",
        "[Warm-up](../README.md#warm-up)",
    ),
    (
        "`emit_clocks`, every model's",
        "[Labels that arrive late](../README.md#labels-that-arrive-late), and the table below",
    ),
    (
        "`ew_cov`'s extra `stats`",
        "its own [section](../README.md#ew_cov--exponentially-weighted-moments)",
    ),
]
CLOCK_FIELDS: list[tuple[str, str, str]] = [
    (
        "scored_clock",
        "the clock the row was scored at, in the clock column's own type; with no clock, the "
        "row's index in its group",
        "never",
    ),
    (
        "learned_clock",
        "the clock of the newest row the model had learned from when this row was scored, in "
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
    ("`i32`", "`Int32`"),
    ("`i64`", "`Int64`"),
    ("`bool`", "`Boolean`"),
    ("`str`", "`String`"),
    ("`list[f64]`", "`List(Float64)`"),
    ("`enum`", "an `Enum` of `withheld_reason`'s three values"),
    ("`clock`", "the clock column's own type"),
]


READING = f"""\
Each field below comes with its dtype, what it holds, and the rows on which
it is null.

## Reading this page

### Field names

Each table lists the fields of the model's plainest spec, whose target is
`y` and whose features are `x0`, and `x1` where the model needs two. Your
own spec's fields carry your column names in their place. The names follow
the grammar in the README's [Output field
names](../README.md#output-field-names): `<stat>_<column>` for a per-column
value, `_<a>_<b>` for a pair, and a suffix where a grid makes several of the
same field.

{table(("in a name", "stands for", "in this page's tables"), NAME_PARTS)}

EW, in a meaning below, is short for exponentially weighted.

### Field types

A field's dtype is the type the bank declares to Polars before it reads a
row, as `po.spec.output_index(spec)` lists it:

{table(("dtype", "the Polars type"), DTYPES)}

### When a field is null

**Every field is null on a skipped row**: a row whose features or weight
hold a null, or a value that counts as one (NaN, ±inf, or a magnitude above
`1e100`). The model learns nothing from such a row. A row of weight 0 is
not skipped: it is scored, and teaches the model nothing.

**Most of a model's own fields are also null on a withheld row**: a row
whose outputs a warm-up gate held back, which `withheld_reason` names
([Warm-up](../README.md#warm-up)). Each table's last column, *also null*,
says where a field is null besides a skipped row, and *withheld* there
means such a row.

### Fields most models write

Four fields appear in nearly every table below, and are defined here once:

{table(FIELD_HEAD, field_rows(SHARED_FIELDS, SHARED_SPEC))}

### What is not listed

The optional outputs are left out: the `emit_*` switches, `conformal`,
`resid_quantiles` and extra `stats`. Each is shown where the README sets it:

{table(("the switch", "where the README shows it"), OPTIONAL)}

`emit_clocks` is the one switch every model takes, and it adds two fields:

{table(FIELD_HEAD, field_rows(CLOCK_FIELDS, CLOCK_SPEC))}

### How this page is made

`scripts/outputs_doc.py` writes this page from each model's plainest spec,
the one `tests/test_model_registry.py` builds. `tests/test_outputs_doc.py`
regenerates it and fails on any difference, so a new field cannot ship
undocumented and a removed one cannot linger. Do not edit this page by
hand: change the generator, then run
`uv run python scripts/outputs_doc.py > docs/OUTPUTS.md`.
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
    for pre in ("filtered_", "predicted_"):
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
    print("# What each model writes\n")
    print(INTRO)
    print("| family | models |")
    print("|---|---|")
    for family, names in FAMILIES.items():
        anchor = family.lower().replace(" ", "-")
        links = " · ".join(f"[`{name}`](#{name})" for name in names)
        print(f"| [{family}](../README.md#{anchor}) | {links} |")
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
            rows = []
            for f, dtype in declared.items():
                s = stem(f, targets, features)
                null = ALSO_NULL_BY_MODEL.get((name, s), ALSO_NULL.get(s, "**undocumented**"))
                rows.append((f"`{f}`", f"`{dtype}`", MEANING.get(s, "**undocumented**"), null))
            print(table(FIELD_HEAD, rows))


if __name__ == "__main__":
    main()
