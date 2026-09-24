"""The workload `scripts/compare_release.py` runs under two builds.

Self-contained, and written against the API every release since 0.8 has:
numeric clocks, and every builder called by keyword. Each spec is built and
run on its own, so a spec a build refuses -- a model or a parameter it does
not have -- is recorded as refused rather than ending the run. Writes every
output field of every spec, and the frame `closed_groups` drains, to a
parquet file, and what ran to a JSON manifest:

    python scripts/release_probe.py OUT.parquet MANIFEST.json
"""

from __future__ import annotations

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
    t[[90, 250]] += 40.0
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


CLOCK = dict(clock="t", max_dclock=6.0, weight="w")
FIT = dict(targets=["y"], features=["x0", "x1", "x2"], group="g", halflife=25.0, **CLOCK)

#: (name, builder, keyword arguments, the specs it reads, if any).
WORKLOAD: list[tuple[str, str, dict, list[str]]] = [
    ("ridge", "ewridge", dict(FIT, min_periods=4.0), []),
    ("ridge_grid", "ewridge", dict(FIT, ridge=[1e-6, 0.5], halflife=[10.0, 60.0]), []),
    (
        "ridge_diag",
        "ewridge",
        dict(
            FIT,
            targets=["y", "z"],
            emit_sigma=True,
            emit_resid_z=True,
            emit_drift=True,
            emit_metrics=True,
            emit_autocorr=True,
            resid_quantiles=[0.1, 0.9],
            conformal=0.9,
        ),
        [],
    ),
    ("ridge_window", "ewridge", dict(FIT, window=40.0), []),
    ("ridge_origin", "ewridge", dict(FIT, add_intercept=False, standardize=False), []),
    (
        "ridge_session",
        "ewridge",
        dict(FIT, session="s", session_gap=10.0, session_shrink=0.5, long_halflife=200.0),
        [],
    ),
    ("ridge_delay", "ewridge", dict(FIT, label_delay=3.0), []),
    ("rls", "rls", dict(FIT), []),
    ("lasso", "lasso", dict(FIT, lasso_path=[0.2, 0.05, 0.0]), []),
    ("enet", "lasso", dict(FIT, lasso_path=[0.1, 0.0], l1_ratio=0.5), []),
    ("kalman", "kalman", dict(FIT, coef_halflife=50.0), []),
    ("huber", "huber", dict(FIT), []),
    ("quantile", "quantile", dict(FIT, quantile=0.8), []),
    ("ftrl", "ftrl", dict(FIT, targets=["hit"]), []),
    ("sgd", "sgd", dict(FIT, learning_rate=0.01), []),
    ("pa", "pa", dict(FIT), []),
    ("holt", "holt", dict(targets=["y"], group="g", halflife=25.0, **CLOCK), []),
    (
        "ew_cov",
        "ew_cov",
        dict(
            features=["x0", "x1", "x2"],
            stats=["mean", "var", "std", "cov", "corr"],
            group="g",
            halflife=25.0,
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
            halflife=25.0,
            **CLOCK,
        ),
        [],
    ),
    (
        "ew_cov_lags",
        "ew_cov",
        dict(features=["x1", "x2"], stats=["corr", "lagcorr"], lags=[1, 3], halflife=25.0, **CLOCK),
        [],
    ),
    ("kmeans", "kmeans", dict(features=["x1", "x2"], k=2, halflife=25.0, **CLOCK), []),
    ("micro", "micro", dict(features=["x1", "x2"], eps=0.5, halflife=25.0, **CLOCK), []),
    (
        "ew_class",
        "ew_class",
        dict(
            features=["x1", "x2"],
            label="lab",
            classes=["down", "up"],
            precision_prior=1.0,
            halflife=25.0,
            **CLOCK,
        ),
        [],
    ),
    ("seqtest", "seqtest", dict(targets=["y"], a="ridge", b="kalman"), ["ridge", "kalman"]),
    (
        "marginal",
        "marginal",
        dict(targets=["y"], features=["x1", "x2"], lags=[1], bins=4, halflife=25.0, **CLOCK),
        [],
    ),
    ("deco", "deco", dict(features=["x0", "x1", "x2"], halflife=25.0, **CLOCK), []),
    ("corrchange", "corrchange", dict(features=["x1", "x2"], span_rows=40, **CLOCK), []),
    ("bocpd", "bocpd", dict(features=["x1", "x2"], prior_scale=[1.0], **CLOCK), []),
    (
        "hmm",
        "hmm",
        dict(features=["x1", "x2"], k=2, precision_prior=0.1, halflife=25.0, **CLOCK),
        [],
    ),
    (
        "rcov",
        "rcov",
        dict(features=["x1", "x2"], group="block", group_close="monotone", block_rows=30),
        [],
    ),
]


def main(out_path: str, manifest_path: str) -> None:
    warnings.simplefilter("ignore")
    df = stream()
    columns: dict[str, pl.Series] = {}
    ran: list[str] = []
    refused: dict[str, str] = {}
    built: dict[str, dict] = {}
    for name, builder, kw, reads in WORKLOAD:
        try:
            built[name] = getattr(po.spec, builder)(name, **kw)
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


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
