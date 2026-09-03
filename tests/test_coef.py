"""`ModelBank.coef()`: the betas behind a fit, readable from a live bank or a
loaded state file, and the two timing facts a user of them needs -- `pred[t]`
is the fit over the rows before t, `coef[t]` the fit after row t.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po


def _grouped(n=300, seed=0):
    rng = np.random.default_rng(seed)
    g = np.where(np.arange(n) % 2 == 0, "a", "b")
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    truth = np.where(g == "a", 1 + 2 * x1 + 3 * x2, -1 - 2 * x1 + 0.5 * x2)
    y = truth + 0.01 * rng.normal(size=n)
    return pl.DataFrame({"g": g, "x1": x1, "x2": x2, "y": y})


GRID = po.spec.ewridge(
    "ols",
    targets=["y"],
    features=["x1", "x2"],
    halflife=[20, 80],
    ridge=0.0,
    group="g",
    coef_every=1,
)


def test_coef_is_the_output_s_last_coef_per_group_and_instance():
    df = _grouped()
    bank = po.ModelBank([GRID])
    out = bank.fit_predict(df)["ols"].struct.unnest()
    c = bank.coef("ols")
    assert c.columns == [
        "group",
        "instance",
        "n_eff",
        "position",
        "target",
        "ridge",
        "feature_set",
        "lambda",
        "term",
        "coef",
    ]
    assert c.select("group", "instance").unique(maintain_order=True).rows() == [
        ("a", "@h20"),
        ("a", "@h80"),
        ("b", "@h20"),
        ("b", "@h80"),
    ]
    for grp in ("a", "b"):
        last = out.filter(df["g"] == grp).tail(1)
        for inst in ("@h20", "@h80"):
            rows = c.filter((pl.col("group") == grp) & (pl.col("instance") == inst))
            assert rows["term"].to_list() == ["intercept", "x1", "x2"]
            assert rows["coef"].to_list() == last[f"coef{inst}"][0].to_list()
            # n_eff is the next row's: the weight after the last update.
            assert rows["n_eff"][0] > last[f"n_eff{inst}"][0]
    # The betas are the truth, per group, and pivot into the usual shape.
    wide = c.pivot("term", index=["group", "instance"], values="coef")
    a = wide.filter(pl.col("group") == "a").row(0, named=True)
    assert abs(a["x1"] - 2) < 0.02 and abs(a["x2"] - 3) < 0.02
    assert c.filter(pl.col("group") == "a").equals(bank.coef("ols", group="a"))


def test_a_loaded_state_file_answers_without_data(tmp_path):
    bank = po.ModelBank([GRID])
    bank.fit_predict(_grouped())
    bank.save(tmp_path / "s.bin")
    assert po.ModelBank.load(tmp_path / "s.bin").coef("ols").equals(bank.coef("ols"))
    assert po.ModelBank.load_bytes(bank.save_bytes()).coef(0).equals(bank.coef("ols"))


def test_coef_does_not_wait_for_min_periods_but_says_so():
    """`pred` waits for `min_periods`; `coef` holds the last solve, whenever
    the schedule ran it, and `n_eff` says how much is behind it."""
    df = _grouped().drop("g")
    early = po.spec.ewridge(
        "m", targets=["y"], features=["x1", "x2"], halflife=30, min_periods=5, solve_every=0
    )
    bank = po.ModelBank([early])
    out = bank.fit_predict(df.head(2))["m"].struct.unnest()
    assert out["pred_y"].is_null().all() and out["coef"][-1] is not None
    c = bank.coef("m")
    assert c["coef"].null_count() == 0 and c["n_eff"][0] < 5
    # Under the default schedule the first row of a stream has not solved.
    lazy = po.spec.ewridge("m", targets=["y"], features=["x1", "x2"], halflife=30)
    bank = po.ModelBank([lazy])
    bank.fit_predict(df.head(1))
    c = bank.coef("m")
    assert c["coef"].is_null().all() and c["term"].to_list() == ["intercept", "x1", "x2"]


def test_mistakes_and_edges_are_named():
    bank = po.ModelBank([GRID])
    bank.fit_predict(_grouped())
    empty = bank.coef("ols", group="never")
    assert empty.shape == (0, 10) and empty.schema == bank.coef("ols").schema
    with pytest.raises(KeyError, match="no spec named 'nope'"):
        bank.coef("nope")
    with pytest.raises(IndexError, match="spec index 3 out of range"):
        bank.coef(3)
    cov = po.ModelBank([po.spec.ew_cov("c", features=["x1", "x2"], halflife=20)])
    with pytest.raises(ValueError, match="ew_cov emits statistics, not coefficients"):
        cov.coef("c")


@pytest.mark.parametrize(
    "spec",
    [
        po.spec.ewridge(
            "m",
            targets=["y", "y2"],
            features=["x1", "x2"],
            halflife=30,
            feature_sets={"one": ["x1"], "all": ["x1", "x2"]},
            ridge=[0.0, 1.0],
        ),
        po.spec.lasso(
            "m", targets=["y"], features=["x1", "x2"], halflife=30, lasso_path=[0.1, 0.0]
        ),
        po.spec.rls("m", targets=["y"], features=["x1", "x2"], halflife=30),
        po.spec.kalman("m", targets=["y"], features=["x1", "x2"], halflife=30, coef_halflife=100),
        po.spec.huber("m", targets=["y"], features=["x1", "x2"], halflife=30),
        po.spec.ftrl("m", targets=["b"], features=["x1", "x2"], halflife=30),
        po.spec.sgd("m", targets=["y"], features=["x1", "x2"], halflife=30),
        po.spec.pa("m", targets=["y"], features=["x1", "x2"], halflife=30),
        po.spec.holt("m", targets=["y"], halflife=30),
    ],
    ids=lambda s: (
        s["model"]["type"]
        + ("+grid" if "feature_sets" in s["model"] and s["model"]["feature_sets"] else "")
    ),
)
def test_every_model_lays_out_as_coef_index(spec):
    rng = np.random.default_rng(1)
    n = 200
    df = pl.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "y": rng.normal(size=n),
            "y2": rng.normal(size=n),
            "b": (rng.random(n) > 0.5).astype(float),
        }
    )
    spec = {**spec, "coef_every": 1}
    bank = po.ModelBank([spec])
    out = bank.fit_predict(df)["m"].struct.unnest()
    c = bank.coef("m")
    layout = po.spec.coef_index(spec)
    assert c.drop("group", "instance", "n_eff", "coef").equals(layout)
    assert c["coef"].to_list() == out["coef"][-1].to_list()


def test_pred_is_the_weighted_least_squares_fit_of_the_rows_before():
    """The use case in one statement: an EW-OLS (`ridge=0`, solved every
    row) predicts row t from the weighted least-squares fit of rows < t,
    weight 0.5**((t-1-i)/halflife) -- a one-sided local regression -- and
    `coef[t]` is the same fit over rows <= t. `bank.coef()` is the last one."""
    rng = np.random.default_rng(0)
    n, h = 200, 20
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    t = np.arange(n)
    y = 0.3 + np.sin(t / 40) * x1 + np.cos(t / 60) * x2 + 0.1 * rng.normal(size=n)
    df = pl.DataFrame({"x1": x1, "x2": x2, "y": y})
    spec = po.spec.ewridge(
        "ols",
        targets=["y"],
        features=["x1", "x2"],
        halflife=h,
        ridge=0.0,
        solve_every=0,
        coef_every=1,
    )
    bank = po.ModelBank([spec])
    out = bank.fit_predict(df)["ols"].struct.unnest()
    X = np.column_stack([np.ones(n), x1, x2])

    def wls(rows, newest):
        w = np.sqrt(0.5 ** ((newest - np.arange(rows)) / h))
        return np.linalg.lstsq(X[:rows] * w[:, None], y[:rows] * w, rcond=None)[0]

    for tt in range(30, n):
        assert abs(X[tt] @ wls(tt, tt - 1) - out["pred_y"][tt]) < 1e-9
        assert np.allclose(wls(tt + 1, tt), out["coef"][tt].to_list(), atol=1e-9)
    assert np.allclose(wls(n, n - 1), bank.coef("ols")["coef"].to_list(), atol=1e-9)
    # The plan streams the same numbers in bounded memory.
    plan = df.lazy().online.fit_predict([spec], chunk_rows=32).collect()["ols"].struct.unnest()
    assert plan["pred_y"].equals(out["pred_y"])
