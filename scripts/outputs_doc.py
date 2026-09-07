"""Generate `docs/OUTPUTS.md`: what each model writes into its struct column.

The *field lists* come from `po.spec.output_fields` on a canonical spec, so
they cannot drift from the code. The *meanings* are written here, once per
field stem, and reviewed like any other prose. `tests/test_outputs_doc.py`
regenerates the document and fails when it differs, so a new field cannot
ship undocumented and a removed one cannot linger.

Run: `uv run python scripts/outputs_doc.py > docs/OUTPUTS.md`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import polars_online as po  # noqa: E402
from test_model_registry import MINIMAL, _build  # noqa: E402

#: One line per field *stem*. `<t>` is a target, `<f>` a feature, `<a>`/`<b>` a
#: pair of columns, `<j>` an index, `<label>` a class label.
MEANING: dict[str, str] = {
    "pred_<t>": "the prediction for `<t>`, computed from the state **before** this row",
    "pred_<f>": "the predictive mean for feature `<f>` under the fitted model",
    "resid_<t>": "`y - pred` for `<t>`; null where the target is null",
    "n_eff": "accumulated weight before this row's update and before its own decay",
    "coef": "the coefficients behind the fit, refreshed on the solve schedule",
    "lam_selected_<t>": "the path point in force for `<t>`, by lowest EW out-of-sample error",
    "mean_<f>": "EW mean of `<f>`",
    "var_<f>": "EW variance of `<f>`",
    "std_<f>": "EW standard deviation of `<f>`",
    "cov_<a>_<b>": "EW covariance of the pair",
    "corr_<a>_<b>": "EW correlation of the pair",
    "partial_corr_<a>_<b>": "the pair's correlation controlling for every other column",
    "mahal": "Mahalanobis distance of the row from the running mean, in standard deviations",
    "cluster": "index of the nearest cluster; null until `warm_rows` have been seen",
    "dist": "distance from the row to that cluster's centre",
    "dist2": "distance to the second-nearest centre, so `dist2 - dist` is the margin",
    "micro": "index of the micro-cluster the row joined",
    "outlier": "true when no micro-cluster was within its radius",
    "n_clusters": "macro-clusters currently linked",
    "n_micro": "live micro-clusters",
    "class": "the most likely class label",
    "p_<label>": "posterior probability of class `<label>`",
    "log_e_pos_<t>": "log of the e-process betting the sign of `<t>` is positive",
    "log_e_neg_<t>": "log of the e-process betting it is negative",
    "n_pos_<t>": "learned rows whose `<t>` was positive",
    "n_neg_<t>": "learned rows whose `<t>` was negative",
    "u": "the equicorrelation the block currently sits at",
    "rho": "the correlation that implies for every pair in the block",
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
    "state": "the most likely state",
}

#: What a model writes when it writes nothing per row.
STATE_ONLY = {
    "marginal": "its product is the state: read the pairs with `ModelBank.marginal()`",
    "rcov": "its product is the closed block: read it from the `group_close` row",
}


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


def main() -> None:
    print("# What each model writes\n")
    print(
        "Every spec adds **one struct column**, named after the spec, and this is\n"
        "what is in it. The names follow the grammar in the README's *Output field\n"
        "names*: `<stat>_<column>` for a per-column value, `_<a>_<b>` for a pair,\n"
        "and a `__l<lambda>` or `__hl<halflife>` suffix where a grid makes several\n"
        "of the same slot. `po.spec.output_fields(spec)` answers the same question\n"
        "at runtime, for the exact spec you built.\n"
    )
    print(
        "Generated by `scripts/outputs_doc.py` from a canonical spec per model and\n"
        "checked by `tests/test_outputs_doc.py` — do not edit by hand. Optional\n"
        "outputs (the `emit_*` switches, `conformal`, `resid_quantiles`, extra\n"
        "`stats`) are not listed here: they are the same for every model that takes\n"
        "them, and the README's *Diagnostics* table has them.\n"
    )
    for name in sorted(MINIMAL):
        spec = _build(name)
        fields = po.spec.output_fields(spec)
        targets = tuple(spec.get("targets") or ())
        features = tuple(spec.get("features") or ())
        print(f"\n## `{name}`\n")
        if name in STATE_ONLY:
            print(f"Nothing per row but `n_eff` — {STATE_ONLY[name]}.\n")
        print("| field | meaning |")
        print("|---|---|")
        for f in fields:
            s = stem(f, targets, features)
            print(f"| `{f}` | {MEANING.get(s, '**undocumented**')} |")


if __name__ == "__main__":
    main()
