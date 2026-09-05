"""`ew_class` -- class-conditional Gaussian classifier (QDA / LDA / naive
Bayes) on `ew_cov`'s per-class moments (docs/PLAN.md 11a, task 27).

Not a regression: the label column is learned from, not predicted as a
number, and the outputs (``class``, one ``p_<class>`` per class) are the
posterior over the classes *as they stood before the row*. Three kinds of
check:

- **The oracle.** ``Oracle`` below replays the model operation for
  operation: one weighted Welford accumulator per class, the same decay of
  the others, the same prior scale, so the class weights, means and
  ``n_eff`` are held **bit for bit** across every covariance shape, null
  labels, null features, weights and an irregular clock. The posteriors go
  through numpy's ``slogdet`` / ``solve`` instead of the core's Cholesky, so
  they are held to a relative ``1e-9``; the class is held exactly.
- **Large data.** Two hundred thousand rows in six dimensions, three
  classes with their own covariances, held to the Bayes rate the generating
  parameters give (the ceiling no classifier can beat) and to calibration;
  the shapes told apart on data that separates them (LDA cannot see a
  spread, naive Bayes cannot see a correlation). No scikit-learn.
- **Edge cases and plumbing.** Everything docs/PLAN.md section 3 promises,
  and every place the model touches the bank: warmup, an unseen class, null
  labels, null features, zero weights, integer and boolean label columns,
  an undeclared label, chunk invariance, save/load, ``predict``, groups, the
  halflife grid, ``coef`` and its index, the expression path, the lazy
  path, the runner, the CLI, and the refusals.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

import polars_online as po
import reference

SHAPES = ["full", "shared", "diagonal"]
NAMES = np.array(["a", "b", "c", "d"])


def gaussians(n, means, covs, seed=0, probs=None):
    """`n` rows from a mixture of Gaussians, with the generating class."""
    rng = np.random.default_rng(seed)
    means = np.asarray(means, float)
    covs = np.asarray(covs, float)
    c = len(means)
    lab = rng.choice(c, size=n, p=probs)
    X = np.empty((n, means.shape[1]))
    for j in range(c):
        idx = np.flatnonzero(lab == j)
        X[idx] = rng.multivariate_normal(means[j], covs[j], size=len(idx))
    return X, lab


def bayes_rate(X, lab, means, covs, probs=None):
    """Accuracy of the Bayes classifier built from the *generating*
    parameters: the ceiling."""
    means = np.asarray(means, float)
    covs = np.asarray(covs, float)
    c = len(means)
    probs = np.full(c, 1.0 / c) if probs is None else np.asarray(probs, float)
    ell = np.empty((len(X), c))
    for j in range(c):
        d = X - means[j]
        q = np.einsum("ij,jk,ik->i", d, np.linalg.inv(covs[j]), d)
        ell[:, j] = math.log(probs[j]) - 0.5 * np.linalg.slogdet(covs[j])[1] - 0.5 * q
    return float(np.mean(ell.argmax(axis=1) == lab))


def frame(X, lab, null_every=None, names=NAMES, **cols):
    df = pl.DataFrame({f"x{i}": X[:, i] for i in range(X.shape[1])})
    y = [str(names[v]) for v in lab]
    if null_every:
        y = [None if i % null_every == 0 else v for i, v in enumerate(y)]
    df = df.with_columns(pl.Series("y", y, pl.String))
    return df.with_columns(**{k: pl.Series(v) for k, v in cols.items()}) if cols else df


def spec(features=("x0", "x1"), classes=("a", "b", "c"), **kw):
    d = dict(
        features=list(features),
        label="y",
        classes=list(classes),
        precision_prior=1.0,
        halflife=200.0,
        min_periods=5.0,
    )
    d.update(kw)
    return po.spec.ew_class("m", **d)


def unnested(out):
    return out.select("m").unnest("m")


# ---------------------------------------------------------------- the oracle


class Oracle:
    """`EwClass` operation for operation: per class an EW weight, mean,
    centered co-moment matrix and prior scale, updated by the weighted
    Welford recursion of `EwCov::update` in the same order of operations,
    the other classes decayed by the same factor. The posteriors are the
    same equations through numpy."""

    def __init__(self, k, n_classes, halflife, min_periods, covariance, prior):
        self.k = k
        self.nc = n_classes
        self.halflife = halflife
        self.min_periods = min_periods
        self.covariance = covariance
        self.prior = prior
        self.n = [0.0] * n_classes
        self.m = [[0.0] * k for _ in range(n_classes)]
        self.c = [[[0.0] * k for _ in range(k)] for _ in range(n_classes)]
        self.ps = [1.0] * n_classes
        self.n_eff = 0.0

    def factor(self, d):
        if math.isinf(self.halflife):
            return 1.0
        return math.exp2(-(d / self.halflife))

    def _update(self, cls, x, lam, w):
        w_new = lam * self.n[cls] + w
        if w_new <= 0.0:
            return
        a = lam * self.n[cls] / w_new
        b = w / w_new
        m, c = self.m[cls], self.c[cls]
        for i in range(self.k):
            di = x[i] - m[i]
            for j in range(self.k):
                dj = x[j] - m[j]
                c[i][j] = a * c[i][j] + a * b * di * dj
        self.ps[cls] = 1.0 if a <= 0.0 else self.ps[cls] * a
        for i in range(self.k):
            m[i] += b * (x[i] - m[i])
        self.n[cls] = w_new

    def score(self, x):
        if self.n_eff < self.min_periods:
            return None
        total = sum(self.n)
        if total <= 0.0:
            return None
        ell = np.full(self.nc, -np.inf)
        seen = [c for c in range(self.nc) if self.n[c] > 0.0]
        x = np.asarray(x, float)
        if self.covariance == "shared":
            M = np.zeros((self.k, self.k))
            for c in seen:
                pi = self.n[c] / total
                M += pi * (np.array(self.c[c]) + self.prior * self.ps[c] * np.eye(self.k))
            logdet = np.linalg.slogdet(M)[1]
            for c in seen:
                d = x - np.array(self.m[c])
                ell[c] = (
                    math.log(self.n[c] / total) - 0.5 * logdet - 0.5 * (d @ np.linalg.solve(M, d))
                )
        else:
            for c in seen:
                ridge = self.prior * self.ps[c]
                d = x - np.array(self.m[c])
                if self.covariance == "full":
                    M = np.array(self.c[c]) + ridge * np.eye(self.k)
                    logdet = np.linalg.slogdet(M)[1]
                    q = d @ np.linalg.solve(M, d)
                else:
                    v = np.maximum(np.diag(np.array(self.c[c])), 0.0) + ridge
                    logdet = float(np.sum(np.log(v)))
                    q = float(np.sum(d * d / v))
                ell[c] = math.log(self.n[c] / total) - 0.5 * logdet - 0.5 * q
        best = int(np.argmax(ell))  # the first maximum
        top = ell[best]
        z = np.exp(ell - top)
        return best, (z / z.sum()).tolist()

    def step(self, x, label, d, w=1.0):
        """`(class, posteriors, n_eff)` for the row, then learn it."""
        lam = self.factor(d)
        n_before = self.n_eff
        out = self.score(x)
        learn = w > 0.0
        for c in range(self.nc):
            if learn and label == c:
                self._update(c, x, lam, w)
            else:
                self.n[c] *= lam
        self.n_eff = lam * self.n_eff + (w if learn else 0.0)
        return out, n_before


def replay(X, lab, *, halflife, min_periods, covariance, prior, t=None, w=None, max_dclock=np.inf):
    """The oracle over a stream: rows with a non-finite feature or weight are
    skipped (their clock delta folds into the next accepted row); a label of
    `None` is scored, not learned from."""
    n, k = X.shape
    nc = int(max(v for v in lab if v is not None)) + 1 if any(v is not None for v in lab) else 1
    o = Oracle(k, nc, halflife, min_periods, covariance, prior)
    d, _ = reference.compute_dclock(t, None, n, max_dclock=max_dclock)
    rows = []
    pending = 0.0
    for i in range(n):
        wi = 1.0 if w is None else w[i]
        if not (np.all(np.isfinite(X[i])) and np.isfinite(wi)):
            pending += d[i]
            rows.append(None)
            continue
        out, n_eff = o.step(X[i].tolist(), lab[i], pending + d[i], wi)
        pending = 0.0
        rows.append((out, n_eff))
    return rows, o


def _same_as_oracle(got: pl.DataFrame, rows, classes, what: str) -> None:
    cls = got["class"].to_list()
    ps = [got[f"p_{c}"].to_list() for c in classes]
    n_eff = got["n_eff"].to_list()
    for i, r in enumerate(rows):
        if r is None:
            assert cls[i] is None and n_eff[i] is None, f"{what}: row {i} not skipped"
            continue
        out, want_n = r
        assert n_eff[i] == want_n, f"{what}: n_eff[{i}] {n_eff[i]!r} vs {want_n!r}"
        if out is None:
            assert cls[i] is None, f"{what}: row {i} should be null"
            assert all(p[i] is None for p in ps), f"{what}: row {i} posteriors should be null"
            continue
        best, post = out
        assert cls[i] == classes[best], f"{what}: class[{i}] {cls[i]} vs {classes[best]}"
        for c, p in enumerate(ps):
            assert p[i] is not None, f"{what}: p_{classes[c]}[{i}] is null"
            assert math.isclose(p[i], post[c], rel_tol=1e-9, abs_tol=0.0), (
                f"{what}: p_{classes[c]}[{i}] {p[i]!r} vs {post[c]!r}"
            )


def _same_means(bank: po.ModelBank, o: Oracle, what: str) -> None:
    got = bank.coef("m")["coef"].to_list()
    want = [v for c in range(o.nc) for v in (o.m[c] if o.n[c] > 0.0 else [None] * o.k)]
    assert got == want, f"{what}: class means"


THREE = dict(
    means=[[0.0, 0.0], [3.0, 0.5], [0.5, 3.0]],
    covs=[[[1.0, 0.3], [0.3, 1.0]], [[0.5, 0.0], [0.0, 2.0]], [[1.5, -0.6], [-0.6, 0.8]]],
)


class TestOracle:
    @pytest.mark.parametrize("covariance", SHAPES)
    def test_every_shape_against_the_replay(self, covariance):
        X, lab = gaussians(1500, seed=1, **THREE)
        df = frame(X, lab)
        params = dict(halflife=150.0, min_periods=3.0, precision_prior=0.7)
        bank = po.ModelBank([spec(covariance=covariance, **params)])
        got = unnested(bank.fit_predict(df))
        rows, o = replay(
            X, lab.tolist(), covariance=covariance, prior=0.7, halflife=150.0, min_periods=3.0
        )
        _same_as_oracle(got, rows, ["a", "b", "c"], covariance)
        _same_means(bank, o, covariance)
        # Posteriors sum to one on every scored row.
        scored = got.filter(pl.col("class").is_not_null())
        total = scored.select(pl.sum_horizontal("p_a", "p_b", "p_c")).to_series()
        assert scored.height > 1400 and (total - 1.0).abs().max() < 1e-12

    @pytest.mark.parametrize("covariance", SHAPES)
    def test_null_labels_null_features_weights_and_an_irregular_clock(self, covariance):
        X, lab = gaussians(900, seed=2, **THREE)
        rng = np.random.default_rng(3)
        X[rng.choice(900, 40, replace=False), 0] = np.nan
        t = np.cumsum(rng.integers(1, 4, 900).astype(float))
        t[300:] += 50.0  # one long gap, capped by max_dclock
        w = rng.uniform(0.2, 2.0, 900)
        w[::17] = 0.0  # zero-weight rows: scored, clock advanced, not learned
        w[::23] = np.nan  # null weight: skipped
        y = [None if i % 11 == 4 else int(v) for i, v in enumerate(lab)]
        df = frame(X, lab, t=t, w=w).with_columns(
            pl.Series("y", [None if v is None else str(NAMES[v]) for v in y], pl.String)
        )
        params = dict(halflife=60.0, min_periods=2.0, precision_prior=0.3)
        s = spec(clock="t", max_dclock=20.0, weight="w", covariance=covariance, **params)
        bank = po.ModelBank([s])
        got = unnested(bank.fit_predict(df))
        rows, o = replay(
            X,
            y,
            covariance=covariance,
            prior=0.3,
            halflife=60.0,
            min_periods=2.0,
            t=t,
            w=w,
            max_dclock=20.0,
        )
        _same_as_oracle(got, rows, ["a", "b", "c"], covariance)
        _same_means(bank, o, covariance)
        # The fixture did what it claims.
        assert df["x0"].is_nan().sum() == 40 and df["y"].null_count() > 0
        assert got["class"].null_count() > 40

    def test_predict_matches_the_oracle_without_learning(self):
        X, lab = gaussians(800, seed=4, **THREE)
        df = frame(X, lab)
        params = dict(halflife=100.0, min_periods=3.0, precision_prior=1.0)
        bank = po.ModelBank([spec(**params)])
        bank.fit_predict(df.slice(0, 500))
        _, o = replay(
            X[:500],
            lab[:500].tolist(),
            covariance="full",
            prior=1.0,
            halflife=100.0,
            min_periods=3.0,
        )
        probe = df.slice(500, 300)
        # With the label column, and without it: predict reads neither.
        for p in (probe, probe.drop("y")):
            got = unnested(bank.predict(p))
            for i, x in enumerate(X[500:].tolist()):
                best, post = o.score(x)
                assert got["class"][i] == "abc"[best]
                assert math.isclose(got["p_a"][i], post[0], rel_tol=1e-9)
                assert got["n_eff"][i] == o.n_eff
        # ... and predicting changed nothing.
        after = unnested(bank.fit_predict(probe))
        rows, _ = replay(
            X, lab.tolist(), covariance="full", prior=1.0, halflife=100.0, min_periods=3.0
        )
        _same_as_oracle(after, rows[500:], ["a", "b", "c"], "after predict")


# ------------------------------------------------------------- large data


SIX = dict(
    means=[
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.5, 0.0, 1.0, 0.0, 0.5, 0.0],
        [0.0, 1.5, 0.0, 1.0, 0.0, 0.5],
    ],
    covs=[
        np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        np.diag([0.5, 2.0, 0.5, 2.0, 0.5, 2.0]),
        np.eye(6) * 0.6 + 0.4,
    ],
)


class TestLargeData:
    def test_two_hundred_thousand_rows_reach_the_bayes_rate_and_are_calibrated(self):
        X, lab = gaussians(200_000, seed=5, probs=[0.5, 0.3, 0.2], **SIX)
        df = frame(X, lab)
        s = spec(
            features=[f"x{i}" for i in range(6)],
            halflife=20_000.0,
            min_periods=50.0,
            precision_prior=0.1,
        )
        got = unnested(po.ModelBank([s]).fit_predict(df))
        truth = np.array(["abc"[v] for v in lab])
        pred = np.array(got["class"].to_list()[100_000:])
        acc = float(np.mean(pred == truth[100_000:]))
        ceiling = bayes_rate(X[100_000:], lab[100_000:], SIX["means"], SIX["covs"], [0.5, 0.3, 0.2])
        assert ceiling > 0.7  # the problem is not trivial
        assert acc >= ceiling - 0.01 and acc <= ceiling + 0.005, (acc, ceiling)
        # Calibration: among the rows whose top posterior is in a bin, the
        # fraction classified right is the bin's mean posterior.
        p = np.column_stack([got[f"p_{c}"].to_numpy()[100_000:] for c in "abc"])
        top = p.max(axis=1)
        right = pred == truth[100_000:]
        for lo in (0.5, 0.6, 0.7, 0.8, 0.9):
            sel = (top >= lo) & (top < lo + 0.1)
            if sel.sum() >= 500:
                assert abs(right[sel].mean() - top[sel].mean()) < 0.03, (lo, sel.sum())
        assert np.abs(p.sum(axis=1) - 1.0).max() < 1e-12

    def test_lda_and_qda_agree_when_the_covariance_is_shared(self):
        cov = [[1.0, 0.4], [0.4, 1.0]]
        p = dict(means=[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], covs=[cov, cov, cov])
        X, lab = gaussians(60_000, seed=6, **p)
        df = frame(X, lab)
        out = {}
        for shape in ("full", "shared"):
            s = spec(halflife=5000.0, min_periods=20.0, covariance=shape, precision_prior=0.1)
            out[shape] = np.array(
                unnested(po.ModelBank([s]).fit_predict(df))["class"].to_list()[30_000:]
            )
        truth = np.array(["abc"[v] for v in lab])[30_000:]
        ceiling = bayes_rate(X[30_000:], lab[30_000:], p["means"], p["covs"])
        for shape in out:
            assert np.mean(out[shape] == truth) >= ceiling - 0.01, shape
        assert np.mean(out["full"] == out["shared"]) > 0.98

    def test_only_qda_sees_a_spread(self):
        # Same mean, different spread: LDA pools the covariances and has
        # nothing left to separate on; QDA and naive Bayes (the features are
        # independent) reach the Bayes rate.
        p = dict(means=[[0.0, 0.0], [0.0, 0.0]], covs=[np.eye(2) * 0.25, np.eye(2) * 4.0])
        X, lab = gaussians(40_000, seed=7, **p)
        df = frame(X, lab)
        truth = np.array(["ab"[v] for v in lab])[20_000:]
        ceiling = bayes_rate(X[20_000:], lab[20_000:], p["means"], p["covs"])
        assert ceiling > 0.8
        acc = {}
        for shape in SHAPES:
            s = spec(classes=("a", "b"), halflife=5000.0, min_periods=20.0, covariance=shape)
            got = unnested(po.ModelBank([s]).fit_predict(df))["class"].to_list()[20_000:]
            acc[shape] = float(np.mean(np.array(got) == truth))
        assert acc["full"] >= ceiling - 0.01 and acc["diagonal"] >= ceiling - 0.01, acc
        assert acc["shared"] < 0.6, acc

    def test_only_a_full_covariance_sees_a_correlation(self):
        # Same marginals, opposite correlations: naive Bayes sees two
        # identical classes; QDA separates them.
        p = dict(
            means=[[0.0, 0.0], [0.0, 0.0]],
            covs=[[[1.0, 0.95], [0.95, 1.0]], [[1.0, -0.95], [-0.95, 1.0]]],
        )
        X, lab = gaussians(40_000, seed=8, **p)
        df = frame(X, lab)
        truth = np.array(["ab"[v] for v in lab])[20_000:]
        ceiling = bayes_rate(X[20_000:], lab[20_000:], p["means"], p["covs"])
        assert ceiling > 0.85
        acc = {}
        for shape in ("full", "diagonal"):
            s = spec(classes=("a", "b"), halflife=5000.0, min_periods=20.0, covariance=shape)
            got = unnested(po.ModelBank([s]).fit_predict(df))["class"].to_list()[20_000:]
            acc[shape] = float(np.mean(np.array(got) == truth))
        assert acc["full"] >= ceiling - 0.01, acc
        assert acc["diagonal"] < 0.6, acc

    def test_a_swap_of_two_classes_is_relearned_within_a_few_halflives(self):
        X, lab = gaussians(30_000, seed=9, **THREE)
        # Halfway, classes a and b trade places.
        lab2 = lab.copy()
        lab2[15_000:] = np.where(lab[15_000:] == 0, 1, np.where(lab[15_000:] == 1, 0, 2))
        df = frame(X, lab2)
        s = spec(halflife=500.0, min_periods=20.0)
        got = np.array(unnested(po.ModelBank([s]).fit_predict(df))["class"].to_list())
        truth = np.array(["abc"[v] for v in lab2])
        before = np.mean(got[10_000:15_000] == truth[10_000:15_000])
        just_after = np.mean(got[15_000:15_500] == truth[15_000:15_500])
        later = np.mean(got[18_000:] == truth[18_000:])
        assert before > 0.9 and later > 0.9, (before, later)
        assert just_after < 0.6, just_after


# ------------------------------------------------------------- edge cases


class TestEdgeCases:
    def test_outputs_are_null_until_min_periods_and_until_a_label(self):
        X, lab = gaussians(60, seed=10, **THREE)
        df = frame(X, lab)
        got = unnested(po.ModelBank([spec(min_periods=5.0)]).fit_predict(df))
        # n_eff before rows 0..5 is 0, 1, 1+lam, ... < 5 for the first six rows.
        assert got["class"][:6].null_count() == 6 and got["class"][6] is not None
        assert got["n_eff"][0] == 0.0
        # No label at all: scored rows stay null until the first labelled one.
        y = [None] * 20 + [str(NAMES[v]) for v in lab[20:]]
        df2 = df.with_columns(pl.Series("y", y, pl.String))
        got2 = unnested(po.ModelBank([spec(min_periods=0.0)]).fit_predict(df2))
        assert got2["class"][:21].null_count() == 21 and got2["n_eff"][20] > 19.0
        assert got2["class"][21] == str(NAMES[lab[20]])

    def test_an_unseen_class_has_posterior_zero_and_null_means(self):
        X, lab = gaussians(300, seed=11, **THREE)
        lab = lab % 2  # only a and b ever appear
        df = frame(X, lab)
        bank = po.ModelBank([spec(classes=("a", "b", "c"), coef_every=50)])
        got = unnested(bank.fit_predict(df))
        assert got["p_c"].drop_nulls().to_list() == [0.0] * (300 - got["p_c"].null_count())
        assert set(got["class"].drop_nulls().to_list()) == {"a", "b"}
        coef = got["coef"][-1].to_list()
        assert all(v is not None and math.isfinite(v) for v in coef[:4])
        assert coef[4:] == [None, None]
        assert bank.coef("m")["coef"].to_list()[4:] == [None, None]
        # The class means are the means of the rows labelled with the class.
        idx = po.spec.coef_index(spec(classes=("a", "b", "c")))
        assert idx["target"].to_list() == ["a", "a", "b", "b", "c", "c"]
        assert idx["term"].to_list() == ["x0", "x1"] * 3

    def test_a_null_label_is_scored_not_learned(self):
        X, lab = gaussians(400, seed=12, **THREE)
        full = frame(X, lab)
        holes = frame(X, lab, null_every=5)
        s = spec(halflife=float("inf"), min_periods=1.0)
        a = po.ModelBank([s])
        b = po.ModelBank([s])
        ga = unnested(a.fit_predict(full))
        gb = unnested(b.fit_predict(holes))
        # Every row is scored in both (b's row 1 excepted: row 0 carried no
        # label, so no class had been seen), and n_eff counts every row in both.
        assert ga["class"].null_count() == 1 and gb["class"].null_count() == 2
        assert gb["class"][1] is None and gb["class"][2] is not None
        assert gb["n_eff"].to_list() == ga["n_eff"].to_list()
        # But b's class means are the means over the labelled rows only.
        ya = np.array([str(NAMES[v]) for v in lab])
        keep = np.arange(400) % 5 != 0
        for name in "abc":
            want = X[keep & (ya == name)].mean(axis=0)
            got = b.coef("m").filter(pl.col("target") == name)["coef"].to_numpy()
            assert np.allclose(got, want, rtol=1e-10, atol=1e-12), name
        assert not np.allclose(a.coef("m")["coef"].to_numpy(), b.coef("m")["coef"].to_numpy())

    def test_integer_and_boolean_label_columns_are_read_as_keys(self):
        X, lab = gaussians(300, seed=13, **THREE)
        df = frame(X, lab)
        want = unnested(po.ModelBank([spec()]).fit_predict(df))
        ints = df.with_columns(pl.Series("y", lab).cast(pl.Int32))
        got = unnested(po.ModelBank([spec(classes=("0", "1", "2"))]).fit_predict(ints))
        assert got["p_0"].to_list() == want["p_a"].to_list()
        assert got["class"].to_list() == [
            None if v is None else str("abc".index(v)) for v in want["class"].to_list()
        ]
        bools = df.with_columns((pl.Series("y", lab) == 0).alias("y"))
        gb = unnested(po.ModelBank([spec(classes=("true", "false"))]).fit_predict(bools))
        assert gb["class"].drop_nulls().is_in(["true", "false"]).all()
        as_str = df.with_columns(pl.Series("y", ["true" if v == 0 else "false" for v in lab]))
        assert gb.equals(
            unnested(po.ModelBank([spec(classes=("true", "false"))]).fit_predict(as_str)),
            null_equal=True,
        )
        cats = df.with_columns(pl.col("y").cast(pl.Categorical))
        gc = unnested(po.ModelBank([spec()]).fit_predict(cats))
        assert gc.equals(want, null_equal=True)

    def test_an_undeclared_label_names_the_row_the_value_and_the_classes(self):
        X, lab = gaussians(50, seed=14, **THREE)
        df = frame(X, lab)
        first_c = int(np.flatnonzero(lab == 2)[0])
        with pytest.raises(ValueError) as exc:
            po.ModelBank([spec(classes=("a", "b"))]).fit_predict(df)
        msg = str(exc.value)
        assert f'label column "y" has the value "c" at row {first_c}' in msg
        assert 'not one of the classes ["a", "b"]' in msg
        assert "null the rows that should only be scored" in msg
        # A nested label column is refused like any key.
        nested = df.with_columns(pl.struct("y").alias("y"))
        with pytest.raises(ValueError, match="label column"):
            po.ModelBank([spec()]).fit_predict(nested)

    def test_a_null_feature_row_is_skipped_and_the_clock_still_runs(self):
        X, lab = gaussians(200, seed=15, **THREE)
        X[150, 1] = np.nan
        df = frame(X, lab)
        got = unnested(po.ModelBank([spec(halflife=20.0, min_periods=1.0)]).fit_predict(df))
        assert got["class"][150] is None and got["n_eff"][150] is None
        lam = 0.5 ** (1 / 20)
        assert got["n_eff"][151] == pytest.approx(got["n_eff"][149] * lam + 1.0, rel=1e-12)
        assert got["n_eff"][152] == pytest.approx(got["n_eff"][151] * lam**2 + 1.0, rel=1e-12)

    def test_a_zero_weight_first_row_is_legal(self):
        X, lab = gaussians(100, seed=16, **THREE)
        w = np.ones(100)
        w[0] = 0.0
        w[7] = 0.0
        df = frame(X, lab, w=w)
        got = unnested(po.ModelBank([spec(weight="w", min_periods=1.0)]).fit_predict(df))
        assert got["n_eff"][0] == 0.0 and got["n_eff"][1] == 0.0
        assert got["class"][1] is None  # n_eff still below min_periods
        assert got["p_a"].drop_nulls().is_finite().all()
        assert got["class"][-1] is not None

    def test_a_row_at_the_input_bound_leaves_everything_finite(self):
        X, lab = gaussians(300, seed=17, **THREE)
        X[100] = [1e100, -1e100]  # at the bound: accepted, and it moves a mean
        X[200] = [1e101, 1e101]  # beyond it: skipped like a null
        df = frame(X, lab)
        bank = po.ModelBank([spec(min_periods=1.0)])
        got = unnested(bank.fit_predict(df))
        assert got["class"][100] is not None and got["n_eff"][101] > got["n_eff"][99]
        assert got["class"][200] is None and got["n_eff"][200] is None
        for c in "abc":
            assert got[f"p_{c}"].drop_nulls().is_finite().all()
        assert got["class"][-1] is not None
        assert bank.solve_failures() == {"m": {"": 0}}

    def test_chunk_invariance(self):
        X, lab = gaussians(700, seed=18, **THREE)
        df = frame(X, lab, null_every=9)
        s = spec(min_periods=1.0, covariance="shared")
        one = unnested(po.ModelBank([s]).fit_predict(df))
        for size in (1, 7, 97, 350):
            bank = po.ModelBank([s])
            many = unnested(
                pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, df.height, size)])
            )
            assert one.drop("coef").equals(many.drop("coef"), null_equal=True), size
            has = one["coef"].is_not_null()
            assert one.filter(has)["coef"].equals(many.filter(has)["coef"]), size

    def test_save_load(self, tmp_path):
        X, lab = gaussians(600, seed=19, **THREE)
        df = frame(X, lab)
        s = spec(min_periods=1.0, covariance="diagonal")
        for cut in (3, 100, 500):
            a = po.ModelBank([s])
            a.fit_predict(df.slice(0, cut))
            path = tmp_path / f"c{cut}.state"
            a.save(path)
            b = po.ModelBank.load(path, specs=[s])
            rest = df.slice(cut, df.height - cut)
            assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True), cut

    def test_groups_are_independent(self):
        X, lab = gaussians(800, seed=20, **THREE)
        df = frame(X, lab, g=["p", "q"] * 400)
        s = spec(group="g")
        both = po.ModelBank([s]).fit_predict(df)
        solo = po.ModelBank([s]).fit_predict(df.filter(pl.col("g") == "q"))
        assert unnested(both.filter(pl.col("g") == "q")).equals(unnested(solo), null_equal=True)

    def test_halflife_grid(self):
        X, lab = gaussians(400, seed=21, **THREE)
        s = spec(classes=("a", "b", "c"), halflife=[50.0, 500.0])
        assert po.spec.output_fields(s) == [
            "class@h50",
            "p_a@h50",
            "p_b@h50",
            "p_c@h50",
            "n_eff@h50",
            "coef@h50",
            "class@h500",
            "p_a@h500",
            "p_b@h500",
            "p_c@h500",
            "n_eff@h500",
            "coef@h500",
        ]
        out = unnested(po.ModelBank([s]).fit_predict(frame(X, lab)))
        assert out["class@h50"].dtype == pl.String
        assert out["n_eff@h50"][-1] < out["n_eff@h500"][-1]
        assert out["p_a@h50"].to_list() != out["p_a@h500"].to_list()

    def test_coef_is_the_class_means_on_the_cadence(self):
        X, lab = gaussians(300, seed=22, **THREE)
        s = spec(coef_every=50)
        out = unnested(po.ModelBank([s]).fit_predict(frame(X, lab)))
        coef = out["coef"]
        assert [i for i in range(300) if coef[i] is not None] == sorted({*range(49, 300, 50), 299})
        assert len(coef[299]) == 6

    def test_coef_index_and_unnest_name_the_means(self):
        s = spec(features=("u", "v", "w"), classes=("neg", "pos"), halflife=[10.0, 20.0])
        cf = po.spec.coef_fields(s)
        assert cf["name"].to_list()[:6] == [
            "coef_neg_u@h10",
            "coef_neg_v@h10",
            "coef_neg_w@h10",
            "coef_pos_u@h10",
            "coef_pos_v@h10",
            "coef_pos_w@h10",
        ]
        ci = po.spec.coef_index(s)
        assert ci["target"].to_list() == ["neg"] * 3 + ["pos"] * 3
        assert ci["term"].to_list() == ["u", "v", "w"] * 2
        X, lab = gaussians(200, seed=23, means=[[0, 0, 0], [2, 2, 2]], covs=[np.eye(3), np.eye(3)])
        df = pl.DataFrame({"u": X[:, 0], "v": X[:, 1], "w": X[:, 2]}).with_columns(
            pl.Series("y", [("neg", "pos")[v] for v in lab])
        )
        s1 = spec(features=("u", "v", "w"), classes=("neg", "pos"), halflife=float("inf"))
        bank = po.ModelBank([s1])
        flat = bank.fit_predict(df).online.unnest([s1])
        names = po.spec.coef_fields(s1)["name"].to_list()
        assert set(names) <= set(flat.columns) and "coef" not in flat.columns
        assert "class" in flat.columns and "p_neg" in flat.columns
        c = bank.coef("m")
        assert c["target"].to_list() == ["neg"] * 3 + ["pos"] * 3
        assert c["coef"].to_list() == [flat[n][-1] for n in names]
        assert c["coef"].to_numpy() == pytest.approx(
            np.concatenate([X[lab == 0].mean(axis=0), X[lab == 1].mean(axis=0)]), abs=1e-9
        )

    def test_output_index_declares_the_dtypes(self):
        idx = po.spec.output_index(spec())
        assert idx["field"].to_list() == ["class", "p_a", "p_b", "p_c", "n_eff", "coef"]
        assert idx["kind"].to_list() == ["class", "p", "p", "p", "n_eff", "coef"]
        assert idx["dtype"].to_list() == ["str", "f64", "f64", "f64", "f64", "list[f64]"]
        assert idx["target"].to_list()[:4] == ["y"] * 4
        assert idx["columns"][0].to_list() == ["x0", "x1"]

    def test_expression_equals_bank(self):
        X, lab = gaussians(400, seed=24, **THREE)
        df = frame(X, lab, null_every=7, g=["p", "q"] * 200)
        bank = unnested(po.ModelBank([spec(group="g")]).fit_predict(df))
        with pytest.warns(po.InMemoryExpressionWarning):
            expr = df.select(
                pl.col("y")
                .online.ew_class(
                    ["x0", "x1"],
                    classes=["a", "b", "c"],
                    precision_prior=1.0,
                    halflife=200.0,
                    min_periods=5.0,
                )
                .over("g")
            ).unnest("y")
        assert bank.equals(expr, null_equal=True)

    def test_lazy_path_equals_bank(self):
        X, lab = gaussians(500, seed=25, **THREE)
        df = frame(X, lab, null_every=7)
        s = spec()
        bank = po.ModelBank([s]).fit_predict(df)
        lazy = df.lazy().online.fit_predict([s]).collect()
        assert bank.equals(lazy, null_equal=True)

    def test_the_runner_agrees_with_the_bank(self, tmp_path):
        X, lab = gaussians(500, seed=26, **THREE)
        df = frame(X, lab, null_every=7, g=["p", "q", "r", "s", "t"] * 100)
        s = spec(group="g")
        want = po.ModelBank([s]).fit_predict(df)
        src, dst = tmp_path / "in.parquet", tmp_path / "out.parquet"
        df.write_parquet(src)
        po.run(input=str(src), output=str(dst), specs=[s])
        got = pl.read_parquet(dst)
        assert unnested(want).equals(unnested(got), null_equal=True)


# --------------------------------------------------------------- refusals


class TestRefusals:
    @pytest.mark.parametrize(
        "flag",
        [
            {"emit_sigma": True},
            {"emit_resid_z": True},
            {"emit_metrics": True},
            {"resid_quantiles": [0.5]},
            {"conformal": 0.9},
            {"emit_autocorr": True},
            {"emit_drift": True},
            {"emit_selected": True},
            {"emit_averaged": True},
        ],
        ids=lambda f: next(iter(f)),
    )
    def test_residual_diagnostics_are_refused_by_name(self, flag):
        (name,) = flag
        with pytest.raises(ValueError, match=f"{name} does not apply to ew_class"):
            spec(**flag)

    @pytest.mark.parametrize(
        ("kw", "msg"),
        [
            ({"classes": ["a"]}, "classes must list at least 2 classes"),
            ({"classes": ["a", "b", "a"]}, 'classes lists "a" more than once'),
            ({"classes": ["a", ""]}, "classes must not contain an empty name"),
            ({"covariance": "spherical"}, "unknown ew_class covariance"),
            ({"precision_prior": 0.0}, "precision_prior must be finite and > 0"),
            ({"precision_prior": -1.0}, "precision_prior must be finite and > 0"),
            ({"features": ["x0", "x0"]}, "more than once"),
            ({"features": ["x0", "y"]}, "both a target and a feature"),
        ],
        ids=lambda v: next(iter(v)) if isinstance(v, dict) else v,
    )
    def test_bad_parameters_name_the_parameter(self, kw, msg):
        with pytest.raises(ValueError, match=msg):
            spec(**kw)

    def test_shapes_are_checked_by_name(self):
        def build(**kw):
            d = dict(features=["x0"], label="y", classes=["a", "b"], precision_prior=1.0)
            d.update(kw)
            return po.spec.ew_class("m", halflife=10.0, **d)

        with pytest.raises(TypeError, match="classes must be a list of strs, got str 'ab'"):
            build(classes="ab")
        with pytest.raises(TypeError, match="classes must be a list of strs, got list"):
            build(classes=[0, 1])
        with pytest.raises(TypeError, match="label must be a str, got list"):
            build(label=["y"])
        with pytest.raises(TypeError, match="covariance must be a str, got int"):
            build(covariance=1)
        with pytest.raises(TypeError, match="precision_prior must be a number, got str"):
            build(precision_prior="1")
        with pytest.raises(ValueError, match="precision_prior must be finite"):
            build(precision_prior=float("inf"))

    def test_takes_a_label_not_targets(self):
        with pytest.raises(TypeError, match=r"ew_class\(\) takes `label`, not targets"):
            po.spec.ew_class(
                "m",
                features=["x0"],
                label="y",
                targets=["y"],
                classes=["a", "b"],
                precision_prior=1.0,
                halflife=10.0,
            )
        assert spec()["targets"] == ["y"]

    def test_unpack_says_what_an_ew_class_struct_holds(self):
        X, lab = gaussians(100, seed=27, **THREE)
        out = po.ModelBank([spec()]).fit_predict(frame(X, lab))
        with pytest.raises(TypeError, match="an ew_class struct a class and its posteriors"):
            po.eval.unpack(out, "m")

    def test_the_cli_runs_it(self, tmp_path, online_cli):
        import subprocess

        X, lab = gaussians(300, seed=28, **THREE)
        src = tmp_path / "in.parquet"
        dst = tmp_path / "out.parquet"
        frame(X, lab, null_every=7).write_parquet(src)
        cfg = tmp_path / "bank.toml"
        cfg.write_text(
            "\n".join(
                [
                    f'input = "{src.as_posix()}"',
                    f'output = "{dst.as_posix()}"',
                    "[[specs]]",
                    'name = "m"',
                    'features = ["x0", "x1"]',
                    'targets = ["y"]',
                    "halflife = 200.0",
                    "min_periods = 5.0",
                    "[specs.model]",
                    'type = "ew_class"',
                    'classes = ["a", "b", "c"]',
                    'covariance = "shared"',
                    "precision_prior = 1.0",
                ]
            )
        )
        subprocess.run([str(online_cli), "--config", str(cfg)], check=True, capture_output=True)
        got = unnested(pl.read_parquet(dst))
        want = unnested(
            po.ModelBank([spec(covariance="shared")]).fit_predict(frame(X, lab, null_every=7))
        )
        assert got.equals(want, null_equal=True)
        assert got["class"].dtype == pl.String
