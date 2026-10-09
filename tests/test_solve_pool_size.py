"""A wide fit gives the same bits whatever the bank's pool size (docs/PLAN.md
task 231).

faer's solves and eigendecompositions ran at the size of the rayon pool
they were called in, and from 512 columns their last bits followed it: a
wide `ewridge` reported other predictions and coefficient standard errors
under another `POLARS_ONLINE_MAX_THREADS`, and on a machine with another
core count, and `ew_cov`'s principal components likewise. The pool is
built once per process, from the variable, so each count runs in a child
interpreter of its own, and the children's outputs are compared by digest.
The Rust tests in `crates/online-core/src/solve/pool_tests.rs` and
`ewridge/tests.rs` hold the same per factorization and per fit, and failed
on the build before the fix.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import polars as pl
import pytest

import child

TIER = "extended"
pytestmark = pytest.mark.extended(reason="four child interpreters, each fitting 520 columns")

SCRIPT = """\
import sys, hashlib
import polars as pl
import polars_online as po
df = pl.read_parquet(sys.argv[1])
k = int(sys.argv[2])
features = [f"x{j}" for j in range(k)]
ridge = po.spec.ewridge(
    "r", targets=["y"], features=features, half_life=200.0, ridge=1e-3,
    min_weight=20.0, solve_every=1e9, max_rows_between_solves=25, coef_every=0,
    emit_se_coef=True,
)
cov = po.spec.ew_cov(
    "c", features=features, half_life=200.0, stats=[], pca=2, max_rows_between_pca=60,
)
out = po.ModelBank([ridge, cov]).fit_predict(df)
print(hashlib.sha256(str(out.to_dict(as_series=False)).encode()).hexdigest())
"""


def test_a_wide_fit_is_the_same_whatever_the_pool_size(tmp_path):
    k, n = 520, 240
    rng = np.random.default_rng(1)
    X = 2.0 + rng.standard_normal((n, k))
    df = pl.DataFrame({f"x{j}": X[:, j] for j in range(k)}).with_columns(
        y=pl.Series(X[:, 0] - X[:, -1] + 0.1 * rng.standard_normal(n))
    )
    data = tmp_path / "wide.parquet"
    df.write_parquet(data)
    script = tmp_path / "run.py"
    script.write_text(SCRIPT)
    digests = {}
    for threads in (1, 3, 8, 14):
        res = subprocess.run(
            [sys.executable, str(script), str(data), str(k)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=child.env(POLARS_ONLINE_MAX_THREADS=str(threads)),
            check=True,
        )
        digests[threads] = res.stdout.strip()
    assert len(set(digests.values())) == 1, digests
