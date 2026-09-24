"""Generate `docs/OUTPUTS.md`: what each model writes into its output column.

The *field lists* come from `po.spec.output_fields` on a canonical spec per
model, `MINIMAL` in `tests/test_model_registry.py`, so they cannot drift from
the code. The *meanings* are written here, once per field stem, and reviewed
like any other prose. `FAMILIES` groups the models as the README's model table
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
    "Moments and correlation": ("ew_cov", "marginal", "deco", "rcov"),
    "Clustering and classification": ("kmeans", "micro", "ew_class"),
    "Sequential tests and regimes": ("seqtest", "corrchange", "hmm", "bocpd"),
}

#: Where the four fields most models write are defined in full.
SHARED = "([shared field](#fields-most-models-write))"

#: One line per field *stem*. `<t>` is a target, `<f>` a feature, `<a>`/`<b>` a
#: pair of columns, `<j>` an index, `<label>` a class label.
MEANING: dict[str, str] = {
    "pred_<t>": "the prediction for `<t>`, computed from the state **before** this row",
    "pred_<f>": "the predictive mean for feature `<f>` under the fitted model",
    "resid_<t>": "`y - pred` for `<t>`; null where the target is null",
    "n_eff": f"accumulated weight before this row's update and before its own decay {SHARED}",
    "coef": f"the numbers behind the fit as one list, on the rows `coef_every` fills {SHARED}",
    "settled_frac": (
        f"how far the decay window had filled before this row; null where nothing decays {SHARED}"
    ),
    "withheld_reason": (
        f"why the row's predictions are null, and null where nothing was withheld {SHARED}"
    ),
    "support_coef": (
        "on `coef`'s rows, each coefficient's data share `1 - ridge * (S^-1)_jj`, laid "
        "out like `coef`; the intercept is not a share (null)"
    ),
    "lam_selected_<t>": "the path point in force for `<t>`, by lowest EW out-of-sample error",
    "mean_<f>": "EW mean of `<f>`",
    "var_<f>": "EW variance of `<f>`",
    "std_<f>": "EW standard deviation of `<f>`",
    "cov_<a>_<b>": "EW covariance of the pair",
    "corr_<a>_<b>": "EW correlation of the pair",
    "partial_corr_<a>_<b>": "the pair's correlation controlling for every other column",
    "mahal": "Mahalanobis distance of the row from the running mean, in standard deviations",
    "cluster": (
        "the nearest cluster's label, read before the row is learned from; null while there "
        "is none: before seeding, which waits for `max(warm_rows, k)` learned rows, or "
        "while no micro-cluster is established"
    ),
    "dist": "distance from the row to the centre `cluster` was read from",
    "dist2": "distance to the second-nearest centre, so `dist2 - dist` is the margin",
    "micro": "id of the micro-cluster the row joins, or opens when none can take it",
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
    "loglik": "log-likelihood of the row under the fitted model",
    "stat": "the test statistic for the span; null except on the row one is due",
    "crit": "the critical value `stat` is compared against",
    "flag": "true on the row where `stat` crossed `crit`",
    "since_flag": "learned rows since the last flag",
    "p_change": "`P(run length <= 1)`: the mass sitting on a change at or just before this row",
    "run_mode": "most likely run length *before* this row, so `t - run_mode` dates the regime",
    "run_mean": "posterior mean run length",
    "logscore": "log predictive density of the row under the run-length mixture",
    "p_<j>": "posterior probability of state `<j>` before this row",
    "p1_<j>": "one-step-ahead probability of state `<j>`",
    "state": "the most likely state for this row: the one with the largest `p1_<j>`",
}

#: What a model writes when it writes nothing per row.
STATE_ONLY = {
    "marginal": (
        "`marginal` writes nothing per row but `n_eff`. Its product is the state, and "
        "`ModelBank.marginal()` reads the pairs from it."
    ),
    "rcov": (
        "`rcov` writes nothing per row but `n_eff`. Its product is the closed block, in the "
        "row `ModelBank.closed_groups()` gives when a group closes (`group_close`)."
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
    ("`<t>`", "a target", "`y`"),
    ("`<f>`", "a feature", "`x0`, `x1`"),
    ("`<a>`, `<b>`", "the two columns of a pair", "`x0`, `x1`"),
    ("`<j>`", "a state's index", "`0`, `1`"),
    ("`<label>`", "a class label", "`a`, `b`"),
    ("`__r<ridge>`", "one value of a `ridge` grid", ""),
    ("`__<set>`", "one of the `feature_sets`, with one ridge value", ""),
    ("`__<set>_r<ridge>`", "one of the `feature_sets`, with one value of a `ridge` grid", ""),
    ("`__l<lambda>`", "one value of `lasso_path`", "`__l0.1`, `__l0`"),
    (
        "`@h<halflife>`",
        "one halflife of a grid, at the end of every field's name: `n_eff@h500`, or "
        "`n_eff@h10m` for a duration",
        "",
    ),
]

#: The four fields most models write, in full: the field, what it holds, and
#: where it is null.
SHARED_FIELDS: list[tuple[str, str, str]] = [
    (
        "`n_eff`",
        "the accumulated weight before this row's update and before its own decay",
        "",
    ),
    (
        "`settled_frac`",
        "how far the decay window had filled toward steady state before this row: "
        "`1 - 2^(-T/halflife)`, with `T` the decay time seen so far, so 0.5 at one halflife "
        "and 0.75 at two. `min_settled_frac` gates on it",
        "where nothing decays",
    ),
    (
        "`withheld_reason`",
        "why the row's predictions are null: `below_min_settled_frac`, `below_min_periods` "
        "or `above_max_error_inflation`. That order is their precedence, so the first that "
        "applies is the one named",
        "where nothing was withheld",
    ),
    (
        "`coef`",
        "the numbers behind the fit, as one flat list written after the row's update: a "
        "regression's coefficients, or what a model that is not a regression keeps in their "
        "place, such as its centres or state means. Its builder's docstring lays the list "
        "out. A model that solves on a schedule (`solve_every`) shows its latest solve",
        "on every row but those `coef_every` fills, which by default are each group's last "
        "row in each chunk; and before the model has anything to report, such as a first solve",
    ),
]


def table(head: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> str:
    """A GitHub table, one row per tuple; an empty string is an empty cell."""

    def line(cells: tuple[str, ...]) -> str:
        return "|" + "|".join(f" {c} " if c else " " for c in cells) + "|"

    return "\n".join([line(head), "|" + "---|" * len(head), *map(line, rows)])


READING = f"""\
[Reading this page](#reading-this-page) says how a field's name is built,
defines the four fields most models write, and says what is left out.

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

### Fields most models write

Four fields appear in nearly every table below, and are defined here once:

{table(("field", "what it holds", "null"), SHARED_FIELDS)}

### What is not listed

The optional outputs are left out: the `emit_*` switches, `conformal`,
`resid_quantiles` and extra `stats`. The diagnostics are the same for every
model that takes them, and the README's [Per-row
diagnostics](../README.md#per-row-diagnostics) shows each switch with the
fields it adds. `ew_cov`'s own
[section](../README.md#ew_cov--exponentially-weighted-moments) lists its
`stats`.

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
        for pre in ("pred", "resid", "lam_selected", "log_e_pos", "log_e_neg", "n_pos", "n_neg"):
            if base == f"{pre}_{t}":
                return f"{pre}_<t>"
    for pre in ("corr", "cov", "partial_corr"):
        if base.startswith(pre + "_"):
            return f"{pre}_<a>_<b>"
    if base.startswith("p1_"):
        return "p1_<j>"
    if base.startswith("p_") and base != "p_change":
        return "p_<j>" if base[2:].isdigit() else "p_<label>"
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
            fields = po.spec.output_fields(spec)
            targets = tuple(spec.get("targets") or ())
            features = tuple(spec.get("features") or ())
            print(f"\n## `{name}`\n")
            if name in STATE_ONLY:
                print(f"{STATE_ONLY[name]}\n")
            print("| field | meaning |")
            print("|---|---|")
            for f in fields:
                s = stem(f, targets, features)
                print(f"| `{f}` | {MEANING.get(s, '**undocumented**')} |")


if __name__ == "__main__":
    main()
