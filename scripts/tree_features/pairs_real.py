"""Candidate 3 on real data: screen the pairs of BTCUSDT's 9 features and of
flights' 7 numeric/code features on the training rows; then an ewridge with
the top-k pair products against without, and XGBoost constrained to them."""

import glob
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings

warnings.filterwarnings("ignore")
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import xgboost as xgb  # noqa: E402
from pairs import anova, binned, edges_of, tables  # noqa: E402

import polars_online as po  # noqa: E402


def resid_additive(x, y, e, B, clock, hl):
    b = binned(x, e)
    cols = {
        f"b{j}_{k}": (b[:, j] == k).astype(float) for j in range(x.shape[1]) for k in range(1, B)
    }
    d = pl.DataFrame({"c": clock, "y": y, **cols})
    s = po.spec.ewridge(
        "o", targets=["y"], features=list(cols), clock="c", gap_cap=600.0, half_life=hl, ridge=1e-3
    )
    p = po.ModelBank([s]).fit_predict(d)["o"].struct.field("pred_y").to_numpy()
    return y - np.nan_to_num(p, nan=np.nanmean(p))


def lin(x, y, clock, te, pairs, hl):
    P = x.shape[1]
    cols = (
        {f"x{i}": x[:, i] for i in range(P)}
        | {f"q{i}": x[:, i] ** 2 for i in range(P)}
        | {f"p{i}_{j}": x[:, i] * x[:, j] for i, j in pairs}
    )
    d = pl.DataFrame({"c": clock, "y": y, **cols})
    s = po.spec.ewridge(
        "o",
        targets=["y"],
        features=list(cols),
        clock="c",
        gap_cap=600.0,
        half_life=hl,
        ridge=1e-3,
        standardize=True,
    )
    p = po.ModelBank([s]).fit_predict(d)["o"].struct.field("pred_y").to_numpy()
    ok = te & ~np.isnan(p)
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2)))


def xg(x, y, tr, va, te, cons):
    p = {"tree_method": "hist", "max_depth": 6, "eta": 0.05, "nthread": 4, "seed": 1}
    if cons is not None:
        p["interaction_constraints"] = str(cons)
    b = xgb.train(
        p,
        xgb.DMatrix(x[tr], y[tr]),
        3000,
        evals=[(xgb.DMatrix(x[va], y[va]), "v")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    return float(
        np.sqrt(
            np.mean(
                (b.predict(xgb.DMatrix(x[te]), iteration_range=(0, b.best_iteration + 1)) - y[te])
                ** 2
            )
        )
    )


def study(name, x, y, clock, tr, va, te, names, hl):
    B = 8
    e = edges_of(x[tr], 5000, B)
    b = binned(x, e)
    r = resid_additive(x, y, e, B, clock, hl)
    T = tables(b[tr], r[tr], B)
    sc = {k: anova(v) for k, v in T.items()}
    ranked = sorted(sc, key=lambda k: -sc[k])
    P = x.shape[1]
    out = {
        "top pairs": [(names[i], names[j], round(sc[(i, j)], 1)) for i, j in ranked[:6]],
        "F of the 10th pair": round(sc[ranked[9]], 1),
        "median F": round(float(np.median(list(sc.values()))), 2),
    }
    out["ewridge x, x^2"] = lin(x, y, clock, te, [], hl)
    for k in (3, 6):
        out[f"ewridge + top-{k} products"] = lin(x, y, clock, te, ranked[:k], hl)
    out["ewridge + all products"] = lin(
        x, y, clock, te, list(itertools.combinations(range(P), 2)), hl
    )
    cons = lambda prs: (  # noqa: E731
        [list(p) for p in prs] + [[i] for i in range(P) if all(i not in p for p in prs)]
    )
    out["xgb unconstrained"] = xg(x, y, tr, va, te, None)
    out["xgb top-6 pairs"] = xg(x, y, tr, va, te, cons(ranked[:6]))
    out["xgb no interactions"] = xg(x, y, tr, va, te, cons([]))
    print(name, json.dumps(out, indent=1), flush=True)
    return out


print(C.load_line())
res = {}
fs = sorted(glob.glob(str(C.SLOT / ".cache" / "BTCUSDT-1m-2024-*.parquet")))[:60]
d = (
    pl.concat([pl.read_parquet(f) for f in fs])
    .sort("t")
    .with_columns(
        z=pl.col("n_trades").log1p(),
        r=(pl.col("close") / pl.col("open")).log().abs(),
        mod=((pl.col("t") % 86400) / 60).cast(pl.Float64),
        wd=pl.from_epoch(pl.col("t").cast(pl.Int64), "s").dt.weekday().cast(pl.Float64),
    )
)
d = d.with_columns(
    [pl.col("z").shift(1).alias("z1")]
    + [pl.col("z").ewm_mean(half_life=h).shift(1).alias(f"zm{h}") for h in (5, 30, 240, 1440)]
    + [pl.col("r").shift(1).alias("r1"), pl.col("r").ewm_mean(half_life=30).shift(1).alias("rm30")]
).slice(1)
F = ["z1", "zm5", "zm30", "zm240", "zm1440", "r1", "rm30", "mod", "wd"]
x = d.select(F).to_numpy()
y = d["z"].to_numpy()
clock = np.arange(len(y), dtype=float)
day = ((d["t"] - d["t"][0]) // 86400).to_numpy()
res["btc"] = study("btc", x, y, clock, day < 40, (day >= 40) & (day < 45), day >= 45, F, 2880.0)
df = C.load_flights()
t = df["t"].to_numpy()
G = ["dep_min", "arr_min", "wday", "distance"]
x = np.column_stack(
    [df[c].to_numpy() for c in G] + [C.codes(df[c]).astype(float) for c in ("carrier", "origin")]
)
res["flights"] = study(
    "flights",
    x,
    df["y"].to_numpy(),
    t,
    t < 243 * 1440.0,
    (t >= 243 * 1440.0) & (t < 273 * 1440.0),
    t >= 273 * 1440.0,
    G + ["carrier", "origin"],
    30 * 1440.0,
)
Path(str(C.OUT / "pairs_real.json")).write_text(json.dumps(res, indent=1), encoding="utf-8")
print(C.load_line())
