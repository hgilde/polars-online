"""Candidate 5 with no build: a LightGBM model's leaves as one-hot columns,
added chunk by chunk with LazyFrame.map_batches(streamable=True), into a
bank's sgd through the IO plugin; compared with the same columns made on the
collected frame. The planted drift of leafrefit.py, seed 0."""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from leafrefit import synth, train_trees  # noqa: E402

import polars_online as po  # noqa: E402

clock, X, y = synth(0)
S = clock.max() + 1
tr, va = clock < 0.4 * S, (clock >= 0.4 * S) & (clock < 0.5 * S)
b = train_trees(X, y, tr, va, 15, 300)
M = b.best_iteration
L = 15
names = [f"x{j}" for j in range(X.shape[1])]
onehot = [f"o{m}_{leaf}" for m in range(M) for leaf in range(L)]


def add_leaves(df: pl.DataFrame) -> pl.DataFrame:
    leaves = b.predict(df.select(names).to_numpy(), pred_leaf=True, num_iteration=M)
    oh = np.zeros((df.height, M * L))
    oh[np.arange(df.height)[:, None], np.arange(M) * L + leaves] = 1.0
    return df.hstack(pl.DataFrame(oh, schema=onehot))


schema = {
    "c": pl.Float64,
    "y": pl.Float64,
    **dict.fromkeys(names, pl.Float64),
    **dict.fromkeys(onehot, pl.Float64),
}
spec = po.spec.sgd(
    "o",
    targets=["y"],
    features=onehot,
    clock="c",
    gap_cap=10.0,
    half_life=20.0,
    schedule="adagrad",
    learning_rate=0.05,
    standardize=False,
)
with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "rows.parquet"
    pl.DataFrame({"c": clock, "y": y, **{n: X[:, j] for j, n in enumerate(names)}}).write_parquet(
        path, row_group_size=10_000
    )
    t0 = time.perf_counter()
    lazy = (
        pl.scan_parquet(path)
        .map_batches(add_leaves, streamable=True, schema=schema)
        .online.fit_predict([spec])
        .select(p=pl.col("o").struct.field("pred_y"))
        .collect()
    )
    t_lazy = time.perf_counter() - t0
    eager = (
        po.ModelBank([spec])
        .fit_predict(add_leaves(pl.read_parquet(path)))["o"]
        .struct.field("pred_y")
    )
print(
    f"streamed through map_batches: {t_lazy:.2f}s; largest difference from the eager run: "
    f"{float((lazy['p'] - eager).abs().max()):.3g}"
)
te = clock >= 0.5 * S
p = lazy["p"].to_numpy()
rmse = np.sqrt(np.nanmean((p[te] - y[te]) ** 2))
print(f"test RMSE {rmse:.4f} (leafrefit2.py's sgd on one-hot: 1.0250)")
