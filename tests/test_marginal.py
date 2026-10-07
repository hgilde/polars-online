"""E44 (Task 37): `marginal` -- every (feature, target) pair's EW moments,
kept in the state and read back as a frame.

Not a regression: nothing is emitted per row but `weight_sum`. The pairs are the
product, so this file holds `ModelBank.marginal()` to a from-scratch numpy
oracle, to `ew_cov` (a pair is the two-column `ew_cov`, to the bit) and to the
shared contract -- chunk invariance, save/load, groups, the half-life grid,
null and zero-weight rows -- on every surface that runs a spec.
"""

import subprocess

import numpy as np
import polars as pl
import pytest

import polars_online as po

PAIR_FIELDS = [
    "weight_sum",
    "n_kish",
    "mean_x",
    "var_x",
    "mean_y",
    "var_y",
    "cov",
    "corr",
    "beta",
    "t_stat",
]
COLUMNS = ["group", "instance", "feature", "target", *PAIR_FIELDS]


def frame(
    n: int = 400,
    p: int = 3,
    m: int = 2,
    seed: int = 0,
    null_feature_every: int | None = None,
    null_target_every: int | None = None,
    weights: bool = False,
    clock: bool = False,
    groups: list[str] | None = None,
) -> pl.DataFrame:
    """`p` features `x<j>`, `m` targets `y<t>` that load on them, and the
    optional plumbing columns `w`, `t` (with gaps) and `g`."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    beta = rng.standard_normal((p, m))
    y = x @ beta + 0.5 * rng.standard_normal((n, m))
    cols: dict[str, object] = {f"x{j}": x[:, j] for j in range(p)}
    for t in range(m):
        cols[f"y{t}"] = y[:, t]
    if null_feature_every:
        cols["x1"] = [None if i % null_feature_every == 3 else v for i, v in enumerate(x[:, 1])]
    if null_target_every:
        cols["y1"] = [None if i % null_target_every == 2 else v for i, v in enumerate(y[:, 1])]
    if weights:
        # 0, 0.5, 1, 1.5, 2 -- zero-weight rows included.
        cols["w"] = 0.5 * (np.arange(n) % 5)
    if clock:
        # Steps of 1 with a gap of 40 every 50 rows, against a cap of 10.
        cols["t"] = np.cumsum(np.where(np.arange(n) % 50 == 0, 40.0, 1.0))
    if groups:
        cols["g"] = [groups[i % len(groups)] for i in range(n)]
    return pl.DataFrame(cols)


def spec(name: str = "m", targets=("y0", "y1"), features=("x0", "x1", "x2"), **kw) -> dict:
    d: dict = dict(targets=list(targets), features=list(features), half_life=60.0)
    d.update(kw)
    return po.spec.marginal(name, **d)


def oracle(
    df: pl.DataFrame, s: dict, upto: int | None = None
) -> tuple[dict[tuple[str, str], dict[str, float]], list[float]]:
    """Every pair's moments after the rows `[:upto]`, from scratch: the
    effective weight of a row is its weight times the decay of every later
    processed row, over the rows where the target is present; plus the
    struct's `weight_sum` for every row -- the accumulated weight before it."""
    rows = df.head(upto) if upto is not None else df
    lam_of = lambda d: 0.5 ** (d / s["half_life"])  # noqa: E731
    cap = s["gap_cap"] if s["gap_cap"] is not None else np.inf
    features, targets = s["features"], s["targets"]
    processed: list[int] = []
    lam: list[float] = []
    w: list[float] = []
    prev_clock = None
    pending = 0.0
    n_eff_col: list[float | None] = []
    w_sum = 0.0
    for i, row in enumerate(rows.iter_rows(named=True)):
        if s["clock"]:
            d = 0.0 if prev_clock is None else min(row["t"] - prev_clock, cap)
            prev_clock = row["t"]
        else:
            d = 0.0 if i == 0 else 1.0
        if any(row[f] is None for f in features):
            pending += d
            n_eff_col.append(None)
            continue
        lam_i = lam_of(min(pending + d, cap))  # the folded total is capped too (S3)
        pending = 0.0
        processed.append(i)
        lam.append(lam_i)
        w.append(row[s["weight"]] if s["weight"] else 1.0)
        n_eff_col.append(w_sum)
        w_sum = lam_i * w_sum + w[-1]
    # Effective weight of processed row k at the end: w_k * prod(lam[k+1:]).
    tail = np.cumprod(np.array(lam[1:] + [1.0])[::-1])[::-1]  # tail[k] = prod(lam[k+1:])
    eff = np.array(w) * tail
    out = {}
    for t in targets:
        yv = np.array([rows[t][i] for i in processed], dtype=float)
        present = ~np.isnan(yv)
        e = np.where(present, eff, 0.0)
        W = e.sum()
        Q = (e**2).sum()
        min_weight = s["min_weight"] if s["min_weight"] is not None else 3.0
        for f in features:
            xv = np.array([rows[f][i] for i in processed], dtype=float)
            xv = np.where(present, xv, 0.0)
            yy = np.where(present, yv, 0.0)
            if W > 0:
                mx, my = (e * xv).sum() / W, (e * yy).sum() / W
                vx = (e * (xv - mx) ** 2).sum() / W
                vy = (e * (yy - my) ** 2).sum() / W
                cov = (e * (xv - mx) * (yy - my)).sum() / W
            else:
                mx = my = vx = vy = cov = 0.0
            n_kish = W * W / Q if Q > 0 else None
            corr = beta = tstat = None
            if min_weight <= W:
                den = np.sqrt(vx) * np.sqrt(vy)
                corr = float(np.clip(cov / den, -1.0, 1.0)) if den > 0 else None
                beta = cov / vx if vx > 0 else None
                if corr is not None and n_kish is not None and n_kish > 2:
                    tstat = corr * np.sqrt((n_kish - 2) / (1 - corr * corr))
            out[(f, t)] = dict(
                weight_sum=W,
                n_kish=n_kish,
                mean_x=mx,
                var_x=vx,
                mean_y=my,
                var_y=vy,
                cov=cov,
                corr=corr,
                beta=beta,
                t_stat=tstat,
            )
    return out, n_eff_col


def close(a, b, tol=1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol * (1.0 + abs(b))


class TestArithmetic:
    def test_matches_a_from_scratch_oracle_with_weights_nulls_gaps_and_zero_weights(self):
        df = frame(
            n=500, null_feature_every=23, null_target_every=7, weights=True, clock=True, seed=1
        )
        s = spec(weight="w", clock="t", gap_cap=10.0, half_life=40.0, min_weight=5.0)
        bank = po.ModelBank([s])
        out = bank.fit_predict(df)
        want, weight_sum = oracle(df, s)
        got = bank.marginal("m")
        assert got.columns == COLUMNS
        assert got.height == 6
        for row in got.iter_rows(named=True):
            pair = want[(row["feature"], row["target"])]
            for field in PAIR_FIELDS:
                assert close(row[field], pair[field]), (row["feature"], row["target"], field)
        # The struct's weight_sum is the weight before each row, every row.
        struct = out["m"].struct.field("weight_sum").to_list()
        assert all(close(a, b) for a, b in zip(struct, weight_sum, strict=True))
        # And the target with nulls has less weight behind it than the other.
        n_eff_by_target = dict(zip(got["target"], got["weight_sum"], strict=True))
        assert n_eff_by_target["y1"] < n_eff_by_target["y0"]
        assert got.select(pl.col(PAIR_FIELDS).is_nan().any()).sum_horizontal().item() == 0

    def test_a_pair_is_the_ew_cov_of_its_two_columns_to_the_bit(self):
        df = frame(n=600, weights=True, clock=True, seed=2)
        common = dict(weight="w", clock="t", gap_cap=10.0, half_life=40.0, min_weight=5.0)
        # `ew_cov` reports before each row; the pair is read after the last
        # row, so it is compared with `ew_cov`'s last-row value over one row more.
        bank = po.ModelBank([spec(**common)])
        bank.fit_predict(df.head(-1))
        got = bank.marginal("m")
        for f in ("x0", "x1", "x2"):
            for t in ("y0", "y1"):
                cov = po.spec.ew_cov(
                    "c", features=[f, t], stats=["mean", "var", "cov", "corr"], **common
                )
                last = po.ModelBank([cov]).fit_predict(df)["c"].to_list()[-1]
                pair = got.filter((pl.col("feature") == f) & (pl.col("target") == t)).row(
                    0, named=True
                )
                assert pair["corr"] == last[f"corr_{f}_{t}"]
                assert pair["mean_x"] == last[f"mean_{f}"]
                assert pair["mean_y"] == last[f"mean_{t}"]
                assert pair["var_x"] == last[f"var_{f}"]
                assert pair["var_y"] == last[f"var_{t}"]
                assert pair["cov"] == last[f"cov_{f}_{t}"]

    def test_kish_size_of_unit_weights_tends_to_the_closed_form(self):
        df = frame(n=3000, seed=3)
        bank = po.ModelBank([spec(half_life=20.0)])
        bank.fit_predict(df)
        lam = 0.5 ** (1.0 / 20.0)
        got = bank.marginal("m")
        assert got["n_kish"].to_numpy() == pytest.approx((1 + lam) / (1 - lam), rel=1e-9)
        assert got["weight_sum"].to_numpy() == pytest.approx(1 / (1 - lam), rel=1e-9)

    def test_min_periods_gates_each_targets_derived_values_by_its_own_weight(self):
        # y1 is present on three rows only: below a min_weight of 5, its
        # correlation, slope and t are null while y0's are numbers, and the
        # moments it has are reported all the same.
        y1 = [None] * 40
        y1[10], y1[20], y1[30] = 1.0, 2.0, 3.0
        df = frame(n=40, seed=4).with_columns(y1=pl.Series(y1))
        bank = po.ModelBank([spec(half_life=float("inf"), min_weight=5.0)])
        bank.fit_predict(df)
        got = bank.marginal("m")
        y0 = got.filter(pl.col("target") == "y0")
        y1_ = got.filter(pl.col("target") == "y1")
        assert y0["corr"].null_count() == 0 and y0["t_stat"].null_count() == 0
        assert y1_["weight_sum"].to_list() == [3.0] * 3 and y1_["n_kish"].to_list() == [3.0] * 3
        assert y1_["mean_y"].to_list() == [2.0] * 3
        assert y1_.select("corr", "beta", "t_stat").null_count().sum_horizontal().item() == 9
        # A per-target list applies per target.
        bank = po.ModelBank([spec(half_life=float("inf"), min_weight=[5.0, 3.0])])
        bank.fit_predict(df)
        y1_ = bank.marginal("m").filter(pl.col("target") == "y1")
        assert y1_["corr"].null_count() == 0

    def test_a_constant_feature_has_null_corr_and_beta_and_a_two_row_target_no_t(self):
        df = frame(n=50, seed=5).with_columns(x2=pl.lit(1.0))
        bank = po.ModelBank([spec(half_life=float("inf"))])
        bank.fit_predict(df)
        got = bank.marginal("m").filter(pl.col("feature") == "x2")
        assert got["var_x"].to_list() == [0.0, 0.0]
        assert got["cov"].to_list() == [0.0, 0.0]
        assert got.select("corr", "beta", "t_stat").null_count().sum_horizontal().item() == 6
        # Two rows: n_kish = 2, so the t-statistic is undefined, and the
        # correlation is +-1 by construction -- which is why min_weight
        # defaults to 3.
        bank = po.ModelBank([spec(half_life=float("inf"), min_weight=2.0)])
        bank.fit_predict(df.head(2))
        two = bank.marginal("m").filter(pl.col("feature") == "x0")
        assert [abs(c) for c in two["corr"].to_list()] == pytest.approx([1.0, 1.0])
        assert two["t_stat"].null_count() == 2
        for rows, nulls in ((2, 2), (3, 0)):
            bank = po.ModelBank([spec(half_life=float("inf"))])
            bank.fit_predict(df.head(rows))
            x0 = bank.marginal("m").filter(pl.col("feature") == "x0")
            assert x0["corr"].null_count() == nulls, rows


class TestPlumbing:
    def test_every_surface_runs_it(self, tmp_path, online_cli):
        df = frame(n=500, null_target_every=7, weights=True, clock=True, seed=10)
        s = spec(weight="w", clock="t", gap_cap=10.0, half_life=40.0)
        ref = po.ModelBank([s])
        one = ref.fit_predict(df).select("m").unnest("m")
        pairs = ref.marginal("m")
        # Lazy plan.
        lazy = df.lazy().online.fit_predict([s]).collect().select("m").unnest("m")
        assert lazy.equals(one, null_equal=True)
        # The state the bank saves carries the pairs back.
        src, state = tmp_path / "in.parquet", tmp_path / "s.state"
        df.write_parquet(src)
        ref.save(state)
        assert po.ModelBank.load(state).marginal("m").equals(pairs, null_equal=True)
        # The CLI, from TOML.
        cli_dst, cli_state = tmp_path / "cli.parquet", tmp_path / "cli.state"
        cfg = tmp_path / "bank.toml"
        cfg.write_text(
            "\n".join(
                [
                    f'input = "{src.as_posix()}"',
                    f'output = "{cli_dst.as_posix()}"',
                    f'save_state = "{cli_state.as_posix()}"',
                    "chunk_rows = 100",
                    "[[specs]]",
                    'name = "m"',
                    'targets = ["y0", "y1"]',
                    'features = ["x0", "x1", "x2"]',
                    'clock = "t"',
                    "gap_cap = 10.0",
                    "half_life = 40.0",
                    'weight = "w"',
                    "[specs.model]",
                    'type = "marginal"',
                ]
            )
        )
        subprocess.run([str(online_cli), "--config", str(cfg)], check=True, capture_output=True)
        assert pl.read_parquet(cli_dst).select("m").unnest("m").equals(one, null_equal=True)
        assert po.ModelBank.load(cli_state).marginal("m").equals(pairs, null_equal=True)

    def test_the_struct_holds_n_eff_alone(self):
        s = spec()
        assert po.spec.output_fields(s) == ["weight_sum"]
        idx = po.spec.output_index(s)
        assert idx["kind"].to_list() == ["weight_sum"] and idx["dtype"].to_list() == ["f64"]
        assert po.spec.coef_fields(s).height == 0
        out = po.ModelBank([s]).fit_predict(frame(n=20))
        assert out.schema["m"] == pl.Struct({"weight_sum": pl.Float64})

    def test_chunk_invariance_to_the_bit(self):
        df = frame(n=700, null_feature_every=23, null_target_every=7, weights=True, clock=True)
        s = spec(weight="w", clock="t", gap_cap=10.0)
        ref = po.ModelBank([s])
        one = ref.fit_predict(df).select("m").unnest("m")
        for size in (1, 31, 250):
            bank = po.ModelBank([s])
            parts = [bank.fit_predict(df.slice(i, size)) for i in range(0, df.height, size)]
            many = pl.concat(parts)
            assert many.select("m").unnest("m").equals(one, null_equal=True)
            assert bank.marginal("m").equals(ref.marginal("m"), null_equal=True), size

    def test_save_load_continues_and_reports_the_same_pairs(self, tmp_path):
        df = frame(n=600, null_target_every=7, weights=True, clock=True)
        s = spec(weight="w", clock="t", gap_cap=10.0)
        a = po.ModelBank([s])
        a.fit_predict(df.slice(0, 300))
        path = tmp_path / "m.state"
        a.save(path)
        b = po.ModelBank.load(path, specs=[s])
        assert b.marginal("m").equals(a.marginal("m"), null_equal=True)
        rest = df.slice(300, 300)
        assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True)
        assert b.marginal("m").equals(a.marginal("m"), null_equal=True)
        # A file written on any OS loads here: the pairs travel in the state.
        c = po.ModelBank.load_bytes(a.save_bytes())
        assert c.marginal("m").equals(a.marginal("m"), null_equal=True)

    def test_groups_and_the_halflife_grid(self):
        df = frame(n=600, groups=["q", "p"], seed=6)
        s = spec(group="g", half_life=[20.0, 200.0], features=["x0", "x1"])
        assert po.spec.output_fields(s) == ["weight_sum@h20", "weight_sum@h200"]
        bank = po.ModelBank([s])
        bank.fit_predict(df)
        got = bank.marginal("m")
        assert got.height == 2 * 2 * 2 * 2
        assert got["group"].to_list() == ["p"] * 8 + ["q"] * 8, "sorted by group"
        assert got["instance"].to_list() == (["@h20"] * 4 + ["@h200"] * 4) * 2
        assert got["target"].to_list() == (["y0", "y0", "y1", "y1"] * 2) * 2, "targets outer"
        assert got["feature"].to_list() == ["x0", "x1"] * 8, "features inner, in spec order"
        # Each group's pairs are what a bank over that group alone reports.
        for g in ("p", "q"):
            solo = po.ModelBank([s])
            solo.fit_predict(df.filter(pl.col("g") == g))
            assert bank.marginal("m", group=g).equals(solo.marginal("m"), null_equal=True)
        assert bank.marginal("m", group="never").height == 0
        assert bank.marginal("m", group="never").columns == COLUMNS
        # The short half-life remembers less.
        short = got.filter(pl.col("instance") == "@h20")["weight_sum"].to_list()
        long = got.filter(pl.col("instance") == "@h200")["weight_sum"].to_list()
        assert all(a < b for a, b in zip(short, long, strict=True))

    def test_zero_weight_rows_learn_nothing_and_keep_the_state_finite(self):
        df = frame(n=30, seed=7).with_columns(w=pl.lit(0.0))
        bank = po.ModelBank([spec(weight="w")])
        out = bank.fit_predict(df)
        assert out["m"].struct.field("weight_sum").to_list() == [0.0] * 30
        got = bank.marginal("m")
        assert got["weight_sum"].to_list() == [0.0] * 6
        assert got.select(pl.col(PAIR_FIELDS).is_nan().any()).sum_horizontal().item() == 0
        assert (
            got.select("n_kish", "corr", "beta", "t_stat").null_count().sum_horizontal().item()
            == 24
        )
        # A zero-weight first row, then real rows: the zero row left no trace.
        w = [0.0] + [1.0] * 29
        with_zero = po.ModelBank([spec(weight="w")])
        with_zero.fit_predict(df.with_columns(w=pl.Series(w)))
        without = po.ModelBank([spec()])
        without.fit_predict(df.slice(1))
        assert with_zero.marginal("m").equals(without.marginal("m"), null_equal=True)

    def test_predict_moves_nothing(self):
        df = frame(n=200, seed=8)
        bank = po.ModelBank([spec()])
        bank.fit_predict(df.head(100))
        before = bank.marginal("m")
        scored = bank.predict(df.tail(100))
        assert scored["m"].struct.field("weight_sum").n_unique() == 1
        assert bank.marginal("m").equals(before, null_equal=True)

    def test_describe_and_summary_see_the_pairs_columns(self):
        df = frame(n=100, null_target_every=7, seed=9)
        bank = po.ModelBank([spec()])
        bank.fit_predict(df)
        desc = bank.describe("m")
        assert desc["column"].to_list() == ["x0", "x1", "x2", "y0", "y1"]
        assert desc["role"].to_list() == ["feature"] * 3 + ["target"] * 2
        assert desc.filter(pl.col("column") == "y1")["null_count"].item() > 0
        assert bank.summary("m")["rows_learned"].item() == 100

    def test_refusals(self):
        for flag, value in [
            ("emit_sigma", True),
            ("emit_zscore", True),
            ("emit_metrics", True),
            ("conformal", 0.9),
            ("resid_quantiles", [0.5]),
            ("emit_autocorr", True),
            ("emit_drift", True),
        ]:
            with pytest.raises(ValueError, match=f"{flag} does not apply to marginal"):
                spec(**{flag: value})
        with pytest.raises(ValueError, match="both a target and a feature"):
            spec(targets=["x0"], features=["x0", "x1"])
        with pytest.raises(ValueError, match="features must be non-empty"):
            spec(features=[])
        with pytest.raises(ValueError, match="targets must be non-empty"):
            spec(targets=[])
        with pytest.raises(TypeError):
            spec(stats=["corr"])  # ew_cov's keyword, not this model's
        ridge = po.spec.ewridge("r", targets=["y0"], features=["x0"], half_life=10.0)
        bank = po.ModelBank([ridge, spec()])
        with pytest.raises(ValueError, match=r'spec "r" has model type "ew_ridge", not "marginal"'):
            bank.marginal("r")
        with pytest.raises(KeyError):
            bank.marginal("nope")
        with pytest.raises(IndexError):
            bank.marginal(5)

    def test_determinism(self):
        df = frame(n=800, groups=["p", "q"], seed=11)
        runs = []
        for _ in range(3):
            bank = po.ModelBank([spec(group="g")])
            bank.fit_predict(df)
            runs.append(bank.marginal("m"))
        assert runs[0].equals(runs[1]) and runs[0].equals(runs[2])


def test_a_wide_pair_set_matches_ew_cov_to_the_bit():
    """The pair layout at p = 12 features x m = 3 targets -- 36 pair slots,
    indexed t * p + j -- against `ew_cov` on a sample of pairs (review
    2026-09-18, phase 4)."""
    df = frame(n=600, p=12, m=3, weights=True, clock=True, seed=2)
    common = dict(weight="w", clock="t", gap_cap=10.0, half_life=40.0, min_weight=5.0)
    feats = [f"x{j}" for j in range(12)]
    targs = [f"y{t}" for t in range(3)]
    bank = po.ModelBank([spec(features=feats, targets=targs, **common)])
    bank.fit_predict(df.head(-1))
    got = bank.marginal("m")
    for f, t in (("x0", "y0"), ("x7", "y2"), ("x11", "y1")):
        cov = po.spec.ew_cov("c", features=[f, t], stats=["mean", "var", "cov", "corr"], **common)
        last = po.ModelBank([cov]).fit_predict(df)["c"].to_list()[-1]
        pair = got.filter((pl.col("feature") == f) & (pl.col("target") == t)).row(0, named=True)
        assert pair["corr"] == last[f"corr_{f}_{t}"], (f, t)
        assert pair["var_x"] == last[f"var_{f}"], (f, t)
        assert pair["mean_y"] == last[f"mean_{t}"], (f, t)


class TestSharedFeatureMoments:
    """docs/PLAN.md task 125 (E72): ``feature_moments="shared"`` keeps one
    mean and variance per feature, over every learned row, where the default
    keeps them per pair; each pair keeps its covariance."""

    COMMON = dict(weight="w", clock="t", gap_cap=10.0, half_life=40.0, min_weight=5.0)

    def test_every_pair_is_per_targets_to_the_bit_where_every_target_is_present(self):
        """Zero-weight rows and clock gaps included: each target's mix is then
        the row's, and each target's feature means the shared ones."""
        df = frame(n=600, weights=True, clock=True, seed=5)
        per = po.ModelBank([spec(**self.COMMON)])
        per.fit_predict(df)
        shared = po.ModelBank([spec(feature_moments="shared", **self.COMMON)])
        shared.fit_predict(df)
        assert shared.marginal("m").equals(per.marginal("m"), null_equal=True)

    def test_the_feature_moments_are_the_columns_over_every_learned_row(self):
        """With ``y1`` absent on every third row, a pair's ``mean_x`` and
        ``var_x`` are the feature's own ``ew_cov`` over every learned row, to
        the bit, for both targets; ``"per_target"`` reads ``y1``'s over its
        own rows, and so reports others there."""
        df = frame(n=600, weights=True, clock=True, null_target_every=3, seed=6)
        shared = po.ModelBank([spec(feature_moments="shared", **self.COMMON)])
        shared.fit_predict(df.head(-1))
        per = po.ModelBank([spec(**self.COMMON)])
        per.fit_predict(df.head(-1))
        got, other = shared.marginal("m"), per.marginal("m")
        for f in ("x0", "x1", "x2"):
            # `ew_cov` reports before each row; the pairs are read after the
            # last row, so it runs one row further (as the test above does).
            cov = po.spec.ew_cov("c", features=[f], stats=["mean", "var"], **self.COMMON)
            last = po.ModelBank([cov]).fit_predict(df)["c"].to_list()[-1]
            for t in ("y0", "y1"):
                pick = (pl.col("feature") == f) & (pl.col("target") == t)
                pair = got.filter(pick).row(0, named=True)
                assert pair["mean_x"] == last[f"mean_{f}"], (f, t)
                assert pair["var_x"] == last[f"var_{f}"], (f, t)
                own = other.filter(pick).row(0, named=True)
                assert (own["var_x"] == pair["var_x"]) == (t == "y0"), (f, t)

    def test_it_is_refused_with_a_window_or_another_value(self):
        with pytest.raises(ValueError, match='"shared" takes no window'):
            spec(feature_moments="shared", window_size=50.0)
        with pytest.raises(ValueError, match='must be "per_target" or "shared"'):
            spec(feature_moments="both")
        spec(feature_moments="per_target", window_size=50.0, lags=[1], window_lags=True)
        spec(feature_moments="shared", lags=[1, 3], serial_rule="bartlett")

    def test_with_lags_every_pair_is_per_targets_to_the_bit_where_every_target_is_present(self):
        """The lagged moments, the lead/lag terms and ``n_serial`` included."""
        df = frame(n=600, weights=True, clock=True, seed=7)
        kw = dict(lags=[1, 2, 5], cross_lags=[1], serial_rule="geometric", **self.COMMON)
        per = po.ModelBank([spec(**kw)])
        per.fit_predict(df)
        shared = po.ModelBank([spec(feature_moments="shared", **kw)])
        shared.fit_predict(df)
        assert shared.marginal("m").equals(per.marginal("m"), null_equal=True)
