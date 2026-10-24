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
        }
    ).with_columns(pl.col("x0", "y").fill_nan(None))


#: Where `states` stops: the first half of `stream()`.
HALF = 200

CLOCK = dict(clock="t", gap_cap=6.0, weight="w")
FIT = dict(targets=["y"], features=["x0", "x1", "x2"], group="g", half_life=25.0, **CLOCK)

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
        dict(features=["x1", "x2"], n_components=2, precision_prior=0.1, half_life=25.0, **CLOCK),
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


def main(out_path: str, manifest_path: str) -> None:
    warnings.simplefilter("ignore")
    df = stream()
    columns: dict[str, pl.Series] = {}
    ran: list[str] = []
    refused: dict[str, str] = {}
    built: dict[str, dict] = {}
    for name, builder, kw, reads in WORKLOAD:
        try:
            built[name] = getattr(po.spec, builder)(name, **_speak(builder, kw))
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
            built[name] = getattr(po.spec, builder)(name, **_speak(builder, kw))
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
