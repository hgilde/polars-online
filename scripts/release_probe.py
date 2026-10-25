"""The workload `scripts/compare_release.py` runs under two builds.

Self-contained, and written against the API every release since 0.8 has:
numeric clocks, and every builder called by keyword. Each spec is built and
run on its own, so a spec a build refuses -- a model or a parameter it does
not have -- is recorded as refused rather than ending the run. Writes every
output field of every spec, and the frame `closed_groups` drains, to a
parquet file, and what ran to a JSON manifest:

    python scripts/release_probe.py OUT.parquet MANIFEST.json

With ``--states``, it fits each spec on the first half of the stream and
saves the bank under the spec's name instead, for
``tests/test_released_state.py``: a released build writes the files, and
this build loads them and goes on:

    python scripts/release_probe.py --states DIR MANIFEST.json
"""

from __future__ import annotations

import inspect
import json
import sys
import warnings

import numpy as np
import polars as pl

import polars_online as po

INF = float("inf")


def stream(n: int = 400, seed: int = 7) -> pl.DataFrame:
    """Three interleaved groups on an irregular clock with gaps past the
    cap, a session change halfway, null features (skipped rows), null
    targets (predict-only rows), zero weights, a 0/1 label, a class label,
    and a block key that only increases, for the models that close groups."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.choice([0.5, 1.0, 1.5, 2.0], size=n))
    # Two gaps past the cap. Raising rows 90 and 250 alone, as this did until
    # task 120, stepped the clock back on the row after each, which the
    # removed `"max"` absorbed and the default now refuses.
    t[90:] += 40.0
    t[250:] += 40.0
    x = rng.standard_normal((n, 3))
    y = 0.5 + x @ np.array([1.0, -0.5, 0.0]) + 0.3 * rng.standard_normal(n)
    z = 0.2 * y + x[:, 2] + 0.3 * rng.standard_normal(n)
    w = rng.uniform(0.5, 1.5, n)
    w[rng.choice(n, 15, replace=False)] = 0.0
    x0 = x[:, 0].copy()
    x0[[30, 31, 200]] = np.nan
    yy = y.copy()
    yy[[45, 46, 300]] = np.nan
    # A clock that steps back by 30 at row 150: a restart under
    # `restart_after_step_back = 10`.
    t_back = t.copy()
    t_back[150:] -= 30.0
    return pl.DataFrame(
        {
            "t": t,
            "g": np.array(["a", "b", "c"])[np.arange(n) % 3],
            "s": np.where(np.arange(n) < n // 2, "am", "pm"),
            "block": np.arange(n) // 50,
            "x0": x0,
            "x1": x[:, 1],
            "x2": x[:, 2],
            "y": yy,
            "z": z,
            "w": w,
            "hit": (y > 0.5).astype(float),
            "lab": np.where(y > 0.5, "up", "down"),
            "t_back": t_back,
        }
    ).with_columns(
        pl.col("x0", "y").fill_nan(None),
        # `t` as a time of day: the temporal clock and its duration
        # parameters, on the same steps.
        ts=pl.datetime(2024, 1, 2, 9, 30) + pl.duration(milliseconds=(pl.col("t") * 1000)),
    )


#: Where `states` stops: the first half of `stream()`.
HALF = 200

CLOCK = dict(clock="t", gap_cap=6.0, weight="w")
FIT = dict(targets=["y"], features=["x0", "x1", "x2"], group="g", half_life=25.0, **CLOCK)


class _Formula:
    """A window expression as a target, built from ``po.<op>`` only when its
    spec is: the probe imports under releases that have no such operator, and
    one of those refuses the spec that names it, not the run."""

    def __init__(self, op: str, column: str, *, alias: str, **kw) -> None:
        self.op, self.column, self.alias, self.kw = op, column, alias, kw

    def build(self) -> pl.Expr:
        kw = dict(self.kw)
        if self.op.startswith("ewm"):
            kw.setdefault("half_life", INF)
        return getattr(po, self.op)(self.column, reverse=True, **kw).alias(self.alias)


def _fullest() -> dict[str, dict]:
    """Each regression with every output it can emit, as
    ``tests/test_budgets_memory.py``'s ``fullest`` switches them on."""
    every = dict(
        coef_every=0,
        emit_clocks=True,
        emit_sigma=True,
        emit_zscore=True,
        emit_metrics=True,
        conformal=0.9,
        resid_quantiles=[0.1, 0.9],
        emit_autocorr=True,
        emit_drift=True,
        drift_threshold=20.0,
        emit_calibration=True,
        emit_breaks=True,
        emit_specification=True,
        emit_tails=True,
        emit_feature_health=True,
        emit_averaged=True,
        emit_error_inflation=True,
        emit_se_coef=True,
        emit_robust_se=True,
        emit_influence=True,
        emit_selected=True,
    )
    simple = {"emit_selected", "emit_error_inflation", "emit_se_coef", "emit_robust_se"}
    simple |= {"emit_influence"}
    lacks = {
        "ewridge": {"emit_selected"},
        "rls": {"emit_selected"},
        "lasso": {"emit_error_inflation", "emit_se_coef", "emit_robust_se", "emit_influence"},
        "kalman": {"emit_selected", "emit_robust_se"},
        "huber": simple,
        "quantile": simple,
        "ftrl": simple,
        "sgd": simple,
        "pa": simple,
        "holt": simple | {"emit_feature_health"},
    }
    fit = dict(FIT, targets=["y", "z"])
    base = {
        "ewridge": fit,
        "rls": fit,
        "lasso": dict(fit, penalties=[0.2, 0.05, 0.0]),
        "kalman": dict(fit, coef_half_life=50.0),
        "huber": fit,
        "quantile": dict(fit, quantile=0.8),
        "ftrl": dict(fit, targets=["hit"]),
        "sgd": dict(fit, learning_rate=0.01),
        "pa": fit,
        "holt": dict(targets=["y"], group="g", half_life=25.0, **CLOCK),
    }
    return {
        kind: {**kw, **{k: v for k, v in every.items() if k not in lacks[kind]}}
        for kind, kw in base.items()
    }


def _ew_class_window() -> dict:
    return dict(
        features=["x1", "x2"],
        label="lab",
        classes=["down", "up"],
        precision_prior=1.0,
        half_life=25.0,
        window_size=60.0,
        coef_every=0,
        **CLOCK,
    )


#: (name, builder, keyword arguments, the specs it reads, if any).
WORKLOAD: list[tuple[str, str, dict, list[str]]] = [
    ("ridge", "ewridge", dict(FIT, min_weight=4.0), []),
    ("ridge_grid", "ewridge", dict(FIT, ridge=[1e-6, 0.5], half_life=[10.0, 60.0]), []),
    (
        "ridge_diag",
        "ewridge",
        dict(
            FIT,
            targets=["y", "z"],
            emit_sigma=True,
            emit_zscore=True,
            emit_drift=True,
            # The default every release before task 168 gave a numeric
            # clock, which this build requires be said with a clock.
            drift_threshold=20.0,
            emit_metrics=True,
            emit_autocorr=True,
            resid_quantiles=[0.1, 0.9],
            conformal=0.9,
        ),
        [],
    ),
    ("ridge_window", "ewridge", dict(FIT, window_size=40.0), []),
    ("ridge_origin", "ewridge", dict(FIT, fit_intercept=False, standardize=False), []),
    (
        "ridge_session",
        "ewridge",
        dict(FIT, session="s", session_gap=10.0, session_shrink=0.5, long_half_life=200.0),
        [],
    ),
    ("ridge_delay", "ewridge", dict(FIT, embargo=3.0), []),
    ("rls", "rls", dict(FIT), []),
    ("lasso", "lasso", dict(FIT, penalties=[0.2, 0.05, 0.0]), []),
    ("enet", "lasso", dict(FIT, penalties=[0.1, 0.0], l1_ratio=0.5), []),
    ("kalman", "kalman", dict(FIT, coef_half_life=50.0), []),
    ("huber", "huber", dict(FIT), []),
    # Task 271's scales: the MAD, the default since, with a memory of its
    # own, and the RMS, every release's before it; and sgd's Huber loss.
    # A release without `huber_scale` refuses them: not comparable.
    ("huber_mad", "huber", dict(FIT, huber_scale="mad", huber_scale_half_life=10.0), []),
    ("huber_rms", "huber", dict(FIT, huber_scale="rms"), []),
    (
        "sgd_huber_mad",
        "sgd",
        dict(FIT, learning_rate=0.01, loss="huber", huber_scale="mad"),
        [],
    ),
    ("quantile", "quantile", dict(FIT, quantile=0.8), []),
    ("ftrl", "ftrl", dict(FIT, targets=["hit"]), []),
    ("sgd", "sgd", dict(FIT, learning_rate=0.01), []),
    ("pa", "pa", dict(FIT), []),
    ("holt", "holt", dict(targets=["y"], group="g", half_life=25.0, **CLOCK), []),
    (
        "ew_cov",
        "ew_cov",
        dict(
            features=["x0", "x1", "x2"],
            stats=["mean", "var", "std", "cov", "corr"],
            group="g",
            half_life=25.0,
            **CLOCK,
        ),
        [],
    ),
    (
        "ew_cov_scores",
        "ew_cov",
        dict(
            features=["x1", "x2", "y"],
            stats=["partial_corr", "mahal"],
            precision_prior=1e-6,
            mahal_quantiles=[0.5, 0.99],
            pca=2,
            half_life=25.0,
            **CLOCK,
        ),
        [],
    ),
    (
        "ew_cov_lags",
        "ew_cov",
        dict(
            features=["x1", "x2"], stats=["corr", "lag_corr"], lags=[1, 3], half_life=25.0, **CLOCK
        ),
        [],
    ),
    ("kmeans", "kmeans", dict(features=["x1", "x2"], n_clusters=2, half_life=25.0, **CLOCK), []),
    ("micro", "micro", dict(features=["x1", "x2"], eps=0.5, half_life=25.0, **CLOCK), []),
    (
        "ew_class",
        "ew_class",
        dict(
            features=["x1", "x2"],
            label="lab",
            classes=["down", "up"],
            precision_prior=1.0,
            half_life=25.0,
            **CLOCK,
        ),
        [],
    ),
    ("seqtest", "seqtest", dict(targets=["y"], a="ridge", b="kalman"), ["ridge", "kalman"]),
    (
        "marginal",
        "marginal",
        dict(targets=["y"], features=["x1", "x2"], lags=[1], bins=4, half_life=25.0, **CLOCK),
        [],
    ),
    ("deco", "deco", dict(features=["x0", "x1", "x2"], half_life=25.0, **CLOCK), []),
    ("corrchange", "corrchange", dict(features=["x1", "x2"], window_rows=40, **CLOCK), []),
    ("bocpd", "bocpd", dict(features=["x1", "x2"], covariance_prior=[1.0], **CLOCK), []),
    (
        "hmm",
        "hmm",
        # `warm_rows` given: the default (1,000 rows since task 278) would
        # seed nothing in the probe's 400.
        dict(
            features=["x1", "x2"],
            n_components=2,
            precision_prior=0.1,
            half_life=25.0,
            warm_rows=50,
            **CLOCK,
        ),
        [],
    ),
    (
        "rcov",
        "rcov",
        dict(features=["x1", "x2"], group="block", group_close="monotone", block_rows=30),
        [],
    ),
    # No `weight`: an audit counts every row whatever it weighs.
    (
        "audit",
        "audit",
        dict(columns=["x0", "x1", "y"], group="g", pairs=True, clock="t", gap_cap=6.0),
        [],
    ),
]

#: The switches and paths the workload above never reached (review round 9,
#: J-2), written against the 1.0 API: every output each regression can emit,
#: a temporal clock, the blocked Gram, feature sets and a prior, the other
#: losses and modes, the windows of the other kinds, a formula target, a
#: drift reset, a solve and coefficient cadence, a restart on a step back.
#: A release before 1.0 refuses some of them by a parameter it lacks.
#: `tests/test_release_compare.py` holds the workload to every field name
#: the fullest workload emits and to the paths it names.
REACH: list[tuple[str, str, dict, list[str]]] = [
    # Every output each regression can emit, as the memory budgets'
    # workload switches them on (tests/test_budgets_memory.py).
    *[(f"full_{kind}", kind, kw, []) for kind, kw in _fullest().items()],
    (
        "ew_cov_full",
        "ew_cov",
        dict(
            features=["x0", "x1", "x2"],
            stats=["mean", "var", "std", "cov", "corr", "partial_corr", "mahal"],
            precision_prior=1e-6,
            mahal_quantiles=[0.5, 0.99],
            pca=2,
            group="g",
            half_life=25.0,
            emit_clocks=True,
            **CLOCK,
        ),
        [],
    ),
    # A temporal clock, every clock parameter a duration.
    (
        "ridge_datetime",
        "ewridge",
        dict(
            FIT,
            clock="ts",
            half_life="25s",
            gap_cap="6s",
            coef_every="20s",
            solve_every="2s",
            embargo="3s",
            session="s",
            session_gap="10s",
            emit_clocks=True,
        ),
        [],
    ),
    # The blocked Gram on two threads, the targets' rows apart.
    (
        "ridge_blocked",
        "ewridge",
        dict(
            FIT,
            targets=["y", "z"],
            gram_block_rows=16,
            gram_threads=2,
            target_gaps="pairwise",
        ),
        [],
    ),
    (
        "ridge_sets",
        "ewridge",
        dict(
            FIT,
            feature_sets={"a": ["x0", "x1"], "b": ["x1", "x2"]},
            emit_selected=True,
        ),
        [],
    ),
    ("ridge_prior", "ewridge", dict(FIT, coef_prior=[[0.4, 1.0, -0.4, 0.1]]), []),
    (
        "ridge_reset",
        "ewridge",
        dict(FIT, emit_drift=True, drift_threshold=3.0, drift_action="reset"),
        [],
    ),
    (
        "ridge_cadence",
        "ewridge",
        dict(FIT, solve_every=8.0, max_rows_between_solves=3, max_rows_between_coefs=5),
        [],
    ),
    (
        "ridge_step_back",
        "ewridge",
        dict(FIT, group=None, clock="t_back", restart_after_step_back=10.0),
        [],
    ),
    ("ridge_both", "ewridge", dict(FIT, window_size=40.0, closed="both"), []),
    ("ridge_lam", "ewridge", dict(FIT, half_life=None, lam=0.97), []),
    (
        "kalman_shared",
        "kalman",
        dict(
            FIT,
            targets=["y", "z"],
            revert_half_life=200.0,
            share_p=True,
            q=[1e-4, 1e-3, 1e-3, 1e-3],
            obs_var=0.1,
        ),
        [],
    ),
    ("kalman_standardized", "kalman", dict(FIT, coef_half_life=50.0, standardize=True), []),
    # Standardized with no `obs_var`, a target's prior is sized from its
    # first three observed rows; a `min_weight` of one row's weight lets
    # rows past it arrive before then, which predict nothing and withhold
    # as `below_warm_rows` (1.2.0 predicted 0.0 there: review round 9, B-1).
    (
        "kalman_unsized",
        "kalman",
        dict(FIT, coef_half_life=50.0, standardize=True, min_weight=1.0),
        [],
    ),
    *[
        (f"sgd_{loss}", "sgd", dict(FIT, learning_rate=0.01, loss=loss, **extra), [])
        for loss, extra in (
            ("huber", {"huber_delta": 1.0}),
            ("quantile", {"quantile": 0.8}),
            ("epsilon_insensitive", {"eps": 0.1}),
            ("poisson", {"targets": ["hit"]}),
            ("logistic", {"targets": ["hit"]}),
        )
    ],
    ("sgd_adagrad", "sgd", dict(FIT, learning_rate=0.1, schedule="adagrad"), []),
    ("ftrl_squared", "ftrl", dict(FIT, loss="squared"), []),
    *[(f"pa_{mode}", "pa", dict(FIT, mode=mode, c=0.5), []) for mode in ("pa1", "pa2")],
    (
        "marginal_window",
        "marginal",
        dict(
            targets=["y"],
            features=["x1", "x2"],
            lags=[1, 2],
            window_lags=True,
            window_size=60.0,
            closed="both",
            half_life=25.0,
            **CLOCK,
        ),
        [],
    ),
    (
        "marginal_edges",
        "marginal",
        dict(
            targets=["y"],
            features=["x1", "x2"],
            bin_edges=[[-1.0, 0.0, 1.0], [-0.5, 0.5]],
            half_life=25.0,
            **CLOCK,
        ),
        [],
    ),
    ("lasso_window", "lasso", dict(FIT, penalties=[0.1, 0.0], window_size=40.0), []),
    ("ew_class_window", "ew_class", _ew_class_window(), []),
    # A target that is a window expression over the next 10 clock
    # units, learned once they have passed. Built when the spec is,
    # so a release without the operator refuses the spec alone.
    (
        "ridge_formula",
        "ewridge",
        dict(
            FIT,
            group=None,
            targets=[_Formula("ewm_mean", "y", window_size=10.0, alias="fwd_y")],
            embargo=10.0,
        ),
        [],
    ),
    (
        "rcov_session",
        "rcov",
        dict(
            features=["x1", "x2"],
            group="g",
            session="s",
            group_close="session",
            block_rows=30,
        ),
        [],
    ),
]

WORKLOAD += REACH


#: Task 144's renames, new name -> old, so the probe can run under a released
#: wheel that predates them (``scripts/compare_release.py``,
#: ``tests/test_released_state.py``): the workload is written in the current
#: names, and translated when the installed package lacks them.
_OLD_NAMES = {
    "half_life": "halflife",
    "long_half_life": "long_halflife",
    "coef_half_life": "coef_halflife",
    "revert_half_life": "revert_halflife",
    "select_half_life": "select_halflife",
    "trend_half_life": "trend_halflife",
    "embargo": "label_delay",
    "gap_cap": "max_dclock",
    "window_size": "window",
    "min_weight": "min_periods",
    "emit_zscore": "emit_resid_z",
    "fit_intercept": "add_intercept",
    "max_iter": "max_cd_iters",
    "tol": "cd_tol",
    "reset_on_flag": "reset",
    # Task 196's row counts, which say so in their names.
    "update_every_rows": "update_every",
    "split_merge_every_rows": "split_merge_every",
    "permute_every_rows": "permute_every",
}


def _current_names() -> bool:
    """Whether the installed package takes task 144's names: asked of a
    builder, since a checked builder's signature is ``(*args, **kwargs)``."""
    try:
        po.spec.ewridge("probe", targets=["y"], features=["x"], half_life=10.0)
    except TypeError:
        return False
    return True


#: The names of 2026-10-11, new name -> the 0.13 name, per builder, so the
#: probe runs under a released wheel that predates them (task 244). The
#: current package reads the old ones too, with a warning, but the workload
#: is written in the new.
_OLD_NAMES_IN = {
    "kmeans": {"n_clusters": "k", "init": "seed_rule"},
    "hmm": {"n_components": "k", "init": "seed_rule", "covariance_type": "covariance"},
    "ew_class": {"covariance_type": "covariance"},
    # Task 195's `delta`: `ridge` is still `ewridge`'s, so only `rls` moves.
    "rls": {"delta": "ridge"},
    "lasso": {"penalties": "lasso_path", "snapshot_every": "window_every"},
    "corrchange": {"window_rows": "span_rows"},
    "bocpd": {
        "mean_prior": "prior_mean",
        "mean_precision_prior": "prior_kappa",
        "degrees_of_freedom_prior": "prior_nu",
        "covariance_prior": "prior_scale",
    },
}


#: The words renamed in one builder's parameter, under its 1.0 name, new
#: word -> the 0.13 word (task 244). `bocpd`'s `covariance_type` was its
#: `emission` word, which `_speak` translates on its own.
_OLD_WORDS_IN = {
    "kmeans": {"init": {"k-means++": "kmeanspp"}},
    "hmm": {
        "init": {"k-means++": "kmeanspp"},
        "covariance_type": {"tied": "shared", "diag": "diagonal"},
    },
    "ew_class": {"covariance_type": {"tied": "shared", "diag": "diagonal"}},
}


def _names_of_1_0() -> bool:
    """Whether the installed package takes the names of 2026-10-11."""
    return "n_clusters" in inspect.signature(po.spec.kmeans).parameters


def _speak(builder: str, kw: dict) -> dict:
    """``kw`` in the installed package's own names."""
    if not _names_of_1_0():
        words = _OLD_WORDS_IN.get(builder, {})
        kw = {k: words[k].get(v, v) if k in words else v for k, v in kw.items()}
        if builder == "bocpd" and "covariance_type" in kw:
            shape = kw.pop("covariance_type")
            kw["emission"] = "gaussian" if shape == "full" else shape
        kw = {_OLD_NAMES_IN.get(builder, {}).get(k, k): v for k, v in kw.items()}
    if _current_names():
        return kw
    out: dict = {}
    for key, value in kw.items():
        if key == "restart_after_step_back":
            out["on_clock_reset"] = "reset_state"
            out["min_backwards_jump"] = value
        elif key == "ridge_scale":
            out["ridge_decay"] = value == "sum"
        elif key == "standardize" and builder == "sgd":
            out["scale_features"] = value
        elif key == "standardize" and builder == "pa" and value is False:
            pass  # task 195's default; before it, `pa` read its features raw
        elif key == "stats":
            # Task 196's `lag_corr`, `lagcorr` before it.
            out["stats"] = ["lagcorr" if s == "lag_corr" else s for s in value]
        else:
            out[_OLD_NAMES.get(key, key)] = value
    return out


def build(builder: str, name: str, kw: dict) -> dict:
    """The spec ``builder`` makes of ``kw`` in the installed package's names,
    a formula target built only now (`_Formula`)."""
    kw = {
        k: [t.build() if isinstance(t, _Formula) else t for t in v] if k == "targets" else v
        for k, v in kw.items()
    }
    return getattr(po.spec, builder)(name, **_speak(builder, kw))


def main(out_path: str, manifest_path: str) -> None:
    warnings.simplefilter("ignore")
    df = stream()
    columns: dict[str, pl.Series] = {}
    ran: list[str] = []
    refused: dict[str, str] = {}
    built: dict[str, dict] = {}
    for name, builder, kw, reads in WORKLOAD:
        try:
            built[name] = build(builder, name, kw)
            bank = po.ModelBank([built[r] for r in reads] + [built[name]])
            out = bank.fit_predict(df)
            closed = bank.closed_groups()
        except Exception as e:  # a spec this build does not have
            refused[name] = f"{type(e).__name__}: {e}"[:300]
            continue
        ran.append(name)
        for field in out[name].struct.fields:
            columns[f"{name}.{field}"] = out[name].struct.field(field)
        for col in closed.columns if closed.height else []:
            columns[f"{name}.closed.{col}"] = closed[col].extend_constant(
                None, df.height - closed.height
            )
    pl.DataFrame(columns).write_parquet(out_path)
    manifest = {"version": po.__version__, "file": po.__file__, "ran": ran, "refused": refused}
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)


def states(out_dir: str, manifest_path: str) -> None:
    """Each spec fitted on the first half of `stream()` and saved to
    ``<out_dir>/<name>.state``, with the specs it reads in the same bank. The
    manifest names what was saved and the schema this build writes."""
    warnings.simplefilter("ignore")
    first = stream().head(HALF)
    saved: list[str] = []
    refused: dict[str, str] = {}
    built: dict[str, dict] = {}
    for name, builder, kw, reads in WORKLOAD:
        try:
            built[name] = build(builder, name, kw)
            bank = po.ModelBank([built[r] for r in reads] + [built[name]])
            bank.fit_predict(first)
            bank.save(f"{out_dir}/{name}.state")
        except Exception as e:  # a spec this build does not have
            refused[name] = f"{type(e).__name__}: {e}"[:300]
            continue
        saved.append(name)
    manifest = {
        "version": po.__version__,
        "schema": po.schema_version(),
        "saved": saved,
        "refused": refused,
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "--states":
        states(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2])
