"""The gradient models and ``holt`` on the paths their oracles leave out, held
row by row -- ``pred``, ``resid``, ``n_eff`` and ``coef`` -- to references
written from each builder's docstring:

- ``ftrl``: ``tests/reference.py::ftrl_ref`` already takes several targets,
  zero weights and labels to clamp, and no test gave it any; nor the L1
  under a halflife.
- ``pa``: ``tests/reference_paths.py::pa_ref``. Its oracles were river's
  (no intercept, unit weights, no decay) and a constrained replay; here the
  intercept is inside the norm, a weight below 1 scales the step and one
  above counts as 1, over several targets.
- ``sgd``: ``tests/reference_paths.py::sgd_ref``, every loss under every
  schedule. Only the squared loss had a per-row oracle, and no oracle had a
  weight other than 0 or 1.
- ``holt``: ``tests/reference_paths.py::holt_ref``, on a clock column with
  weights, zero steps, null targets and the per-target gate; statsmodels
  holds it on settled rows at unit weight on a row count.

``sgd``'s ``l2`` and ``clip_gradient`` are left at values that do nothing:
the docstring's ``g_i = d * z_i * w + l2 * b_i`` puts the ridge on the
intercept, which the replay in ``tests/test_constraints.py`` says the core
exempts, and "a cap on the gradient's magnitude" does not say which norm.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from reference import ftrl_ref
from reference_paths import holt_ref, pa_ref, sgd_ref

MAX_DCLOCK = 20.0
FEATURES = ["x0", "x1", "x2"]
TARGETS = ["ya", "yb"]

# Measured over the cases below as |got - expected| / (1 + |expected|): pred
# 8.2e-16, resid 1.2e-15, n_eff exact, coef 4.1e-16. The tolerance is 100x
# the largest, rounded up to a power of ten.
#
# Seeded into copies of the references, each of these fails a case here by
# far more (pred, same measure). ftrl: the decay after the weights 7e-3 to
# 3.5e-2; n decayed inside the root (review C24) 6.9e-2 to 0.35; a zero
# weight learned 8.5e-2 to 0.19; a null target taken as 0, 0.37-1.2; a label
# not clamped 4.7e-2. pa: the intercept outside the norm (river's) 0.23-2.1;
# no cap on a weight above 1, 0.23-0.35; pa2 damped by 1 / c, 0.14. sgd: the
# weight dropped 2.8e-2 to 0.32; inv_scaling on n_eff after the row 3.5e-3
# to 7.4e-2; AdaGrad's G after the step 0.97-1.0 (or diverged), or not
# decayed 8.7e-2 to 0.26; huber unclipped 0.25-0.51; quantile's sign flipped
# 1.4-1.5; the insensitive loss's gradient linear 0.29-0.37. holt: a zero
# step counted toward the trend 1.9e-3; the first observation weighting the
# trend 4.1e-2 to 0.15; the default trend halflife 1x 0.14; extrapolating by
# the row's step 0.14.
TOL = 1e-12


def _stream(seed: int, kind: str = "linear", dup: bool = False, n: int = 400) -> pl.DataFrame:
    """Three features and two targets on an irregular clock with two gaps
    past ``max_dclock``: one target with scattered nulls, one with a block;
    weights between 0.5 and 1.5 and a twentieth of them 0, the first row
    among them; three rows skipped for a null feature. ``dup`` puts 25 rows
    on the clock of the row before them."""
    rng = np.random.default_rng(seed)
    dt = rng.choice([0.5, 1.0, 1.5, 2.0], n)
    dt[0] = 0.0
    dt[[120, 290]] = 50.0
    if dup:
        dt[rng.choice(np.arange(1, n), 25, replace=False)] = 0.0
    t = np.cumsum(dt)
    x = 0.8 * rng.standard_normal((n, 3))
    if kind == "linear":
        ya = 0.5 + x @ np.array([1.0, -0.5, 0.25]) + 0.3 * rng.standard_normal(n)
        yb = -1.0 + x @ np.array([0.0, 0.8, -0.6]) + 0.3 * rng.standard_normal(n)
    elif kind == "poisson":
        ya = rng.poisson(np.exp(0.2 + x @ np.array([0.4, -0.3, 0.1]))).astype(float)
        yb = rng.poisson(np.exp(-0.3 + x @ np.array([0.0, 0.3, -0.2]))).astype(float)
    elif kind == "logistic":
        ya = (rng.random(n) < 1 / (1 + np.exp(-(x @ np.array([1.5, -1.0, 0.5]))))).astype(float)
        yb = (rng.random(n) < 1 / (1 + np.exp(-(0.3 + x @ np.array([0.0, 1.0, -1.0]))))).astype(
            float
        )
        ya[[20, 21, 222]] = [1.5, -0.5, 2.0]  # labels the logistic loss clamps into [0, 1]
    else:  # a level and a trend in the clock, for holt
        ya = 2.0 + 0.05 * t + 0.3 * rng.standard_normal(n)
        yb = -1.0 - 0.02 * t + 0.2 * rng.standard_normal(n)
    ya[rng.choice(n, 30, replace=False)] = np.nan
    yb[200:240] = np.nan
    w = rng.uniform(0.5, 1.5, n)
    w[0] = 0.0
    w[rng.choice(np.arange(1, n), 20, replace=False)] = 0.0
    x1 = x[:, 1].copy()
    x1[[60, 61, 330]] = np.nan
    frame = {"t": t, "x0": x[:, 0], "x1": x1, "x2": x[:, 2], "ya": ya, "yb": yb, "w": w}
    return pl.DataFrame(frame).with_columns(pl.col(c).fill_nan(None) for c in ["x1", *TARGETS])


def _inputs(df):
    dc = np.zeros(df.height)
    dc[1:] = np.diff(df["t"].to_numpy())
    return (
        df.select(FEATURES).to_numpy(),
        df.select(TARGETS).to_numpy().astype(float),
        dc,
        df["w"].to_numpy(),
    )


def _close(got, exp, what):
    assert (np.isnan(got) == np.isnan(exp)).all(), (
        f"{what}: null patterns differ at rows {np.flatnonzero(np.isnan(got) != np.isnan(exp))[:8]}"
    )
    ok = ~np.isnan(exp)
    err = np.abs(got[ok] - exp[ok]) / (1.0 + np.abs(exp[ok]))
    assert err.size == 0 or err.max() <= TOL, f"{what}: max rel diff {err.max():.3e}"


def _held(out, ref, y, coef=True, coef_where=None):
    for j, t in enumerate(TARGETS):
        _close(
            out.struct.field(f"pred_{t}").to_numpy().astype(float), ref["pred"][:, j], f"pred_{t}"
        )
        resid = out.struct.field(f"resid_{t}").to_numpy().astype(float)
        _close(resid, y[:, j] - ref["pred"][:, j], f"resid_{t}")
        assert np.isfinite(ref["pred"][:, j]).sum() > 250, f"{t} is scored too little"
    _close(out.struct.field("n_eff").to_numpy().astype(float), ref["n_eff"], "n_eff")
    if not coef:
        return
    rows = out.struct.field("coef").to_list()
    empty = [np.nan] * ref["coef"][0].size
    got = np.array([empty if r is None else r for r in rows], float).reshape(ref["coef"].shape)
    where = ~np.isnan(ref["n_eff"]) if coef_where is None else coef_where
    _close(got[where], ref["coef"][where], "coef")


class TestFtrl:
    """Two targets with nulls, zero weights and clamped labels."""

    @pytest.mark.parametrize(
        "kw",
        [
            {"halflife": 150.0, "l1": 0.5, "alpha": 0.3, "beta": 0.5, "l2": 0.5},
            {"halflife": 100.0, "l1": 0.2, "alpha": 0.2, "loss": "squared", "add_intercept": False},
            {"halflife": float("inf"), "l1": 1.0},
        ],
        ids=["logistic-l1-under-a-halflife", "squared-no-intercept", "logistic-no-decay"],
    )
    def test_every_row(self, kw):
        df = _stream(1, kind="linear" if kw.get("loss") == "squared" else "logistic")
        x, y, dc, w = _inputs(df)
        spec = po.spec.ftrl(
            "m",
            targets=TARGETS,
            features=FEATURES,
            clock="t",
            max_dclock=MAX_DCLOCK,
            weight="w",
            min_periods=8.0,
            **kw,
        )
        out = po.ModelBank([spec]).fit_predict(df)["m"]
        ref = ftrl_ref(x, y, dc, w, min_periods=8.0, max_dclock=MAX_DCLOCK, **kw)
        # ftrl_ref reports the weights a row was scored with, the bank's
        # `coef` those after it; they meet only without decay (below).
        _held(out, ref, y, coef=False)

    def test_without_decay_coef_is_the_next_rows_weights(self):
        """With nothing to decay, the weights after row ``i`` are the ones
        row ``i + 1`` is scored with."""
        df = _stream(2, kind="logistic")
        x, y, dc, w = _inputs(df)
        spec = po.spec.ftrl(
            "m",
            targets=TARGETS,
            features=FEATURES,
            clock="t",
            max_dclock=MAX_DCLOCK,
            weight="w",
            halflife=float("inf"),
            l1=1.0,
            coef_every=1,
        )
        out = po.ModelBank([spec]).fit_predict(df)["m"]
        ref = ftrl_ref(x, y, dc, w, max_dclock=MAX_DCLOCK, l1=1.0)
        accepted = np.flatnonzero(~np.isnan(ref["n_eff"]))
        rows = out.struct.field("coef").to_list()
        got = np.array([rows[i] for i in accepted[:-1]], float).reshape(-1, 2, 4)
        _close(got, ref["coef"][accepted[1:]], "coef")
        assert (got == 0.0).any(), "the L1 should zero a coefficient"


class TestPassiveAggressive:
    @pytest.mark.parametrize("mode", ["pa", "pa1", "pa2"])
    def test_every_mode(self, mode):
        df = _stream(1)
        x, y, dc, w = _inputs(df)
        kw = {"mode": mode, "c": 0.3, "eps": 0.1, "halflife": 50.0, "min_periods": 5.0}
        spec = po.spec.pa(
            "m",
            targets=TARGETS,
            features=FEATURES,
            clock="t",
            max_dclock=MAX_DCLOCK,
            weight="w",
            coef_every=1,
            **kw,
        )
        out = po.ModelBank([spec]).fit_predict(df)["m"]
        _held(out, pa_ref(x, y, dc, w, max_dclock=MAX_DCLOCK, **kw), y)
        assert (w > 1.0).any() and ((w > 0.0) & (w < 1.0)).any()

    def test_without_an_intercept(self):
        df = _stream(2)
        x, y, dc, w = _inputs(df)
        kw = {"mode": "pa1", "c": 0.5, "eps": 0.05, "halflife": 50.0, "min_periods": 5.0}
        spec = po.spec.pa(
            "m",
            targets=TARGETS,
            features=FEATURES,
            clock="t",
            max_dclock=MAX_DCLOCK,
            weight="w",
            coef_every=1,
            add_intercept=False,
            **kw,
        )
        out = po.ModelBank([spec]).fit_predict(df)["m"]
        _held(out, pa_ref(x, y, dc, w, max_dclock=MAX_DCLOCK, add_intercept=False, **kw), y)


LOSSES = [
    ("squared", "linear", {}),
    ("huber", "linear", {"huber_delta": 0.3}),
    ("quantile", "linear", {"quantile": 0.8}),
    ("epsilon_insensitive", "linear", {"eps": 0.2}),
    ("poisson", "poisson", {}),
    ("logistic", "logistic", {}),
]
SCHEDULES = [
    ("constant", {"learning_rate": 0.05}),
    ("inv_scaling", {"learning_rate": 0.05, "power": 0.25}),
    ("adagrad", {"learning_rate": 0.2}),
]


class TestSgd:
    @pytest.mark.parametrize(("schedule", "rate"), SCHEDULES, ids=[s for s, _ in SCHEDULES])
    @pytest.mark.parametrize(("loss", "kind", "extra"), LOSSES, ids=[s for s, *_ in LOSSES])
    def test_every_loss_under_every_schedule(self, loss, kind, extra, schedule, rate):
        df = _stream(3, kind=kind)
        x, y, dc, w = _inputs(df)
        kw = {"loss": loss, "schedule": schedule, "halflife": 50.0, "min_periods": 5.0}
        kw |= extra | rate
        spec = po.spec.sgd(
            "m",
            targets=TARGETS,
            features=FEATURES,
            clock="t",
            max_dclock=MAX_DCLOCK,
            weight="w",
            coef_every=1,
            **kw,
        )
        out = po.ModelBank([spec]).fit_predict(df)["m"]
        _held(out, sgd_ref(x, y, dc, w, max_dclock=MAX_DCLOCK, **kw), y)

    def test_without_an_intercept(self):
        df = _stream(4)
        x, y, dc, w = _inputs(df)
        kw = {"learning_rate": 0.05, "halflife": 50.0, "min_periods": 5.0}
        spec = po.spec.sgd(
            "m",
            targets=TARGETS,
            features=FEATURES,
            clock="t",
            max_dclock=MAX_DCLOCK,
            weight="w",
            coef_every=1,
            add_intercept=False,
            **kw,
        )
        out = po.ModelBank([spec]).fit_predict(df)["m"]
        _held(out, sgd_ref(x, y, dc, w, max_dclock=MAX_DCLOCK, add_intercept=False, **kw), y)


class TestHolt:
    @pytest.mark.parametrize(
        ("kw", "dup"),
        [
            ({"level_halflife": 20.0, "min_periods": 2.0}, False),
            ({"level_halflife": 20.0, "trend_halflife": 60.0, "min_periods": 12.0}, True),
        ],
        ids=["default-trend-halflife", "zero-steps-and-the-gate"],
    )
    def test_every_row(self, kw, dup):
        """The second case puts 25 rows at the clock of the row before, and
        a ``min_periods`` the gappy target's own weight falls under after
        its block of nulls."""
        df = _stream(5 if not dup else 6, kind="holt", dup=dup)
        y = df.select(TARGETS).to_numpy().astype(float)
        spec = po.spec.holt(
            "m",
            targets=TARGETS,
            clock="t",
            max_dclock=MAX_DCLOCK,
            weight="w",
            coef_every=1,
            **kw,
        )
        out = po.ModelBank([spec]).fit_predict(df)["m"]
        ref = holt_ref(y, df["t"].to_numpy(), df["w"].to_numpy(), max_dclock=MAX_DCLOCK, **kw)
        # Before a target's first observation the reference has no level and
        # the core reports [0, 0]; nothing documents which, so from the
        # first observation on.
        _held(out, ref, y, coef_where=~np.isnan(ref["coef"]).any(axis=(1, 2)))
        if dup:
            assert np.isnan(ref["pred"][240:300, 1]).sum() > 10, "the gate should withhold"
