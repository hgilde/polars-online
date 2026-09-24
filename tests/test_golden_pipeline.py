"""T-W7: fixed numbers out of the whole pipeline, compared on every OS.

`crates/online-core/tests/golden.rs` pins the Rust core's arithmetic, but
nothing pinned what comes out of the *Polars* layer -- extraction, the
per-group fan-out, the diagnostics, the struct assembly. A divergence
introduced there, or by polars' own vectorized paths on a different CPU, would
be invisible until someone compared two machines by hand.

This is that comparison, made automatic: the numbers are committed, and CI runs
this file on ubuntu, macOS and Windows. Locally the agreement is exact; the
tolerance is what "the same answer on another platform" is allowed to mean.

The constants come from the current implementation, which is independently
verified against the numpy oracles in `tests/reference.py` (agreement ~1e-13
for every model) -- the same bargain `golden.rs` makes. Regenerate only after
confirming a change is intended:

    uv run python tests/test_golden_pipeline.py
"""

import numpy as np
import polars as pl

import polars_online as po

#: How far two platforms may disagree, relative. Different LLVM vectorization
#: and BLAS paths can reorder floating-point operations; a genuinely divergent
#: algorithm shows up far above this.
TOL = 1e-12

#: Rows sampled from the stream. Early, mid and late, so warmup, the clock gap
#: and the converged state are all represented.
PICKS = [25, 60, 119]


def stream(n: int = 120) -> pl.DataFrame:
    """One deterministic frame exercising every input path: two groups, an
    irregular clock with a long gap, nulls in a feature and in the target,
    varying weights, and a session break."""
    rng = np.random.default_rng(20260830)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n) * 3.0 + 2.0
    y = 1.5 * x0 - 0.75 * x1 + 0.25 + 0.1 * rng.standard_normal(n)
    t = np.cumsum(np.where(np.arange(n) % 17 == 0, 9.0, 1.0))
    w = 0.5 + 0.5 * (np.arange(n) % 3)

    x1 = [None if i % 31 == 7 else v for i, v in enumerate(x1)]
    y = [None if i % 29 == 11 else v for i, v in enumerate(y)]
    # A class label for `ew_class`: which side of its mean `x0` falls on, with
    # a null every 13th row (scored, not learned from).
    label = [None if i % 13 == 5 else ("hi" if v > 0.0 else "lo") for i, v in enumerate(x0)]
    return pl.DataFrame(
        {
            "t": t,
            "x0": x0,
            "x1": x1,
            "y0": y,
            "label": label,
            "w": w,
            "g": ["a" if i % 2 == 0 else "b" for i in range(n)],
            "session": ["m" if i < n // 2 else "n" for i in range(n)],
        }
    )


INF = float("inf")


def specs() -> list[dict]:
    common = dict(
        targets=["y0"],
        features=["x0", "x1"],
        clock="t",
        max_dclock=6.0,
        halflife=25.0,
        weight="w",
        group="g",
        min_periods=4.0,
    )
    return [
        po.spec.ewridge(
            "ridge",
            ridge=[1e-6, 0.5],
            standardize=True,
            max_rows_between_solves=1,
            emit_sigma=True,
            emit_resid_z=True,
            conformal=0.9,
            **common,
        ),
        po.spec.rls("rls", ridge=1.0, **common),
        po.spec.kalman("kalman", coef_halflife=80.0, **common),
        po.spec.lasso("lasso", lasso_path=[0.2, 0.0], max_rows_between_solves=1, **common),
        po.spec.huber("huber", max_rows_between_solves=1, **common),
        po.spec.quantile("quantile", quantile=0.75, max_rows_between_solves=1, **common),
        po.spec.sgd("sgd", learning_rate=0.02, **common),
        po.spec.pa("pa", **common),
        po.spec.sgd("sgd_simplex", learning_rate=0.02, coef_min=0.0, coef_sum=1.0, **common),
        po.spec.pa("pa_box", coef_min=-0.5, coef_max=0.5, **common),
        po.spec.kalman(
            "kalman_revert", coef_halflife=80.0, revert_halflife=[INF, 30.0, 8.0], **common
        ),
        po.spec.ftrl("ftrl", loss="squared", alpha=0.5, **common),
        po.spec.holt(
            "holt",
            targets=["y0"],
            clock="t",
            max_dclock=6.0,
            halflife=25.0,
            weight="w",
            group="g",
            min_periods=4.0,
        ),
        po.spec.ew_cov(
            "moments",
            features=["x0", "x1"],
            stats=["mean", "var", "corr"],
            clock="t",
            max_dclock=6.0,
            halflife=25.0,
            weight="w",
            group="g",
            min_periods=4.0,
        ),
        po.spec.kmeans(
            "kmeans",
            features=["x0", "x1"],
            k=2,
            warm_rows=8,
            split_merge=0.5,
            split_merge_every=10,
            clock="t",
            max_dclock=6.0,
            halflife=25.0,
            weight="w",
            group="g",
            min_periods=4.0,
        ),
        po.spec.micro(
            "micro",
            features=["x0", "x1"],
            eps=0.5,
            beta_mu=2.0,
            prune_every=10,
            clock="t",
            max_dclock=6.0,
            halflife=25.0,
            weight="w",
            group="g",
            min_periods=4.0,
        ),
        po.spec.ew_class(
            "ew_class",
            features=["x0", "x1"],
            label="label",
            classes=["lo", "hi"],
            covariance="shared",
            precision_prior=0.5,
            clock="t",
            max_dclock=6.0,
            halflife=25.0,
            weight="w",
            group="g",
            min_periods=4.0,
        ),
        # Only `n_eff` per row; the pairs are read from the state at the end.
        po.spec.marginal("marginal", **common),
        # A constancy test over spans of 30 rows, so the 120-row stream
        # closes several per group.
        # A cap above the stream's 9-unit gaps: at `max_dclock = 6` every
        # gap is capped, `clear_lags` abandons the span (task 47), and no
        # span ever closes -- which is correct and pins nothing.
        po.spec.corrchange(
            "corrchange",
            features=["x0", "x1"],
            span_rows=20,
            alpha=0.05,
            clock="t",
            max_dclock=100.0,
            group="g",
        ),
        # A hidden Markov model: the states are given, so the pinned
        # numbers are the filter's and not the seeding's.
        po.spec.hmm(
            "hmm",
            features=["x0", "x1"],
            k=2,
            precision_prior=1e-2,
            learn=True,
            transition=[0.9, 0.1, 0.2, 0.8],
            means=[-1.0, -1.0, 1.0, 1.0],
            covs=[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            clock="t",
            max_dclock=6.0,
            halflife=25.0,
            weight="w",
            group="g",
        ),
        # A block's realised covariance. The stream's groups interleave, so
        # the close is on the session, not monotone; the block itself is
        # pinned below, since nothing but `n_eff` is emitted per row.
        po.spec.rcov(
            "rcov",
            features=["x0", "x1"],
            kind="kernel",
            bandwidth=3,
            clock="t",
            max_dclock=6.0,
            group="g",
            group_close="session",
            session="session",
        ),
        # One number for the whole correlation matrix. The stream has two
        # feature columns, so this is the unblocked form; the block path is
        # pinned by the core golden and by the numpy oracle in test_deco.py.
        po.spec.deco(
            "deco",
            features=["x0", "x1"],
            dynamics="linear",
            alpha=0.05,
            beta=0.9,
            clock="t",
            max_dclock=6.0,
            halflife=25.0,
            weight="w",
            group="g",
            min_periods=4.0,
        ),
        # A posterior over run lengths. Weighted rows scale each run's
        # statistics, and a fractional weight is legal here, so the golden
        # pins the weighted recursion and not just the plain one.
        po.spec.bocpd(
            "bocpd",
            features=["x0", "x1"],
            hazard=40.0,
            emission="diag",
            prior_scale=[1.0],
            prune_below=1e-8,
            max_run=50,
            clock="t",
            max_dclock=6.0,
            weight="w",
            group="g",
            min_periods=4.0,
        ),
        # No weight and no halflife: a test counts trials and does not forget.
        po.spec.seqtest(
            "seqtest", targets=["y0"], clock="t", max_dclock=6.0, group="g", min_periods=4.0
        ),
        # The two-phase bank: reads the ridge grid's `resid_y0__r0.5` and
        # kalman's `resid_y0` from the structs assembled above it.
        po.spec.seqtest(
            "seqtest_compare",
            targets=["y0"],
            a="ridge",
            a_suffix="__r0.5",
            b="kalman",
            clock="t",
            max_dclock=6.0,
            group="g",
            min_periods=4.0,
        ),
    ]


def signature() -> dict[str, float | str | None]:
    """Every non-coefficient output field, at three fixed rows; and, for a
    `marginal`, whose output is its state, every pair's derived values after
    the last row."""
    bank = po.ModelBank(specs())
    out = bank.fit_predict(stream())
    sig: dict[str, float | str | None] = {}
    for spec in specs():
        name = spec["name"]
        for field in po.spec.output_fields(spec):
            if field.startswith(("coef", "support_coef")):
                continue  # a list, and a reporting cadence rather than a value
            values = out[name].struct.field(field).to_list()
            for row in PICKS:
                sig[f"{name}.{field}@{row}"] = values[row]
        if spec["model"]["type"] == "marginal":
            for pair in bank.marginal(name).iter_rows(named=True):
                where = f"[{pair['group']}/{pair['feature']}]@end"
                for field in ("n_eff", "n_kish", "corr", "beta", "t"):
                    sig[f"{name}.{field}{where}"] = pair[field]
        # `corrchange` reports only where a span closes, and the picks
        # above are ordinary rows, so the statistics are pinned in the
        # order they were produced.
        if spec["model"]["type"] == "corrchange":
            stats = out[name].struct.field("stat").drop_nulls().to_list()
            for i, v in enumerate(stats):
                sig[f"{name}.stat#{i}"] = v
    # `rcov` emits nothing per row: its value is the block a group close
    # produces, so that is what is pinned.
    for row in bank.closed_groups(drop=False).iter_rows(named=True):
        where = f"[{row['spec']}/{row['group']}/{row['session']}]"
        sig[f"{row['spec']}.rcov_n{where}"] = row["rcov_n"]
        sig[f"{row['spec']}.bandwidth{where}"] = row["bandwidth_used"]
        for i, v in enumerate(row["rcov"] or []):
            sig[f"{row['spec']}.rcov{i}{where}"] = v
    return sig


#: Produced by `uv run python tests/test_golden_pipeline.py`.
GOLDEN: dict[str, float | str | None] = {
    "ridge.pred_y0__r0.000001@25": -4.684371132566456,
    "ridge.pred_y0__r0.000001@60": -0.2556320797220242,
    "ridge.pred_y0__r0.000001@119": -0.1597673183471509,
    "ridge.resid_y0__r0.000001@25": -0.1672346916682601,
    "ridge.resid_y0__r0.000001@60": -0.21640904909505376,
    "ridge.resid_y0__r0.000001@119": 0.10906854738892208,
    "ridge.pred_y0__r0.5@25": -3.2074053272438663,
    "ridge.pred_y0__r0.5@60": -0.1169734771432751,
    "ridge.pred_y0__r0.5@119": -0.5303952399350174,
    "ridge.resid_y0__r0.5@25": -1.64420049699085,
    "ridge.resid_y0__r0.5@60": -0.35506765167380283,
    "ridge.resid_y0__r0.5@119": 0.4796964689767885,
    "ridge.sigma_y0__r0.000001@25": 0.05149318322426837,
    "ridge.sigma_y0__r0.000001@60": 0.14974398485680726,
    "ridge.sigma_y0__r0.000001@119": 0.09333817729646236,
    "ridge.sigma_y0__r0.5@25": 3.1349106139192906,
    "ridge.sigma_y0__r0.5@60": 0.6856591658858987,
    "ridge.sigma_y0__r0.5@119": 0.854495805542607,
    "ridge.resid_z_y0__r0.000001@25": -3.247705447532783,
    "ridge.resid_z_y0__r0.000001@60": -1.4451936036161652,
    "ridge.resid_z_y0__r0.000001@119": 1.1685309328732296,
    "ridge.resid_z_y0__r0.5@25": -0.5244808224165783,
    "ridge.resid_z_y0__r0.5@60": -0.5178486182927948,
    "ridge.resid_z_y0__r0.5@119": 0.5613795478752293,
    "ridge.lo_y0__r0.000001@25": -4.783126440034132,
    "ridge.lo_y0__r0.000001@60": -0.4257622150611754,
    "ridge.lo_y0__r0.000001@119": -0.2920540148392792,
    "ridge.lo_y0__r0.5@25": -12.11710703850159,
    "ridge.lo_y0__r0.5@60": -0.9867954001987035,
    "ridge.lo_y0__r0.5@119": -9.102917951890893,
    "ridge.hi_y0__r0.000001@25": -4.58561582509878,
    "ridge.hi_y0__r0.000001@60": -0.08550194438287298,
    "ridge.hi_y0__r0.000001@119": -0.027480621855022674,
    "ridge.hi_y0__r0.5@25": 5.702296384013857,
    "ridge.hi_y0__r0.5@60": 0.7528484459121533,
    "ridge.hi_y0__r0.5@119": 8.042127472020857,
    "ridge.coverage_y0__r0.000001@25": 1.0,
    "ridge.coverage_y0__r0.000001@60": 0.708544385155888,
    "ridge.coverage_y0__r0.000001@119": 0.746039777195224,
    "ridge.coverage_y0__r0.5@25": 1.0,
    "ridge.coverage_y0__r0.5@60": 0.8207252289588411,
    "ridge.coverage_y0__r0.5@119": 1.0,
    "ridge.n_eff@25": 7.999488060097996,
    "ridge.n_eff@60": 12.473100285951407,
    "ridge.n_eff@119": 15.110060335371337,
    "ridge.settled_frac@25": 0.5136725262938573,
    "ridge.settled_frac@60": 0.8564127056253706,
    "ridge.settled_frac@119": 0.9782071302132749,
    "ridge.withheld_reason@25": None,
    "ridge.withheld_reason@60": None,
    "ridge.withheld_reason@119": None,
    "rls.pred_y0@25": -4.706391171521188,
    "rls.pred_y0@60": -0.24224285328032447,
    "rls.pred_y0@119": -0.16011065148415327,
    "rls.resid_y0@25": -0.14521465271352785,
    "rls.resid_y0@60": -0.22979827553675347,
    "rls.resid_y0@119": 0.10941188052592443,
    "rls.n_eff@25": 7.999488060097996,
    "rls.n_eff@60": 12.473100285951407,
    "rls.n_eff@119": 15.110060335371337,
    "rls.settled_frac@25": 0.5136725262938573,
    "rls.settled_frac@60": 0.8564127056253706,
    "rls.settled_frac@119": 0.9782071302132749,
    "rls.withheld_reason@25": None,
    "rls.withheld_reason@60": None,
    "rls.withheld_reason@119": None,
    "kalman.pred_y0@25": -2.7312829037141775,
    "kalman.pred_y0@60": -2.96152520455183,
    "kalman.pred_y0@119": -0.21457789071127054,
    "kalman.resid_y0@25": -2.1203229205205387,
    "kalman.resid_y0@60": 2.4894840757347523,
    "kalman.resid_y0@119": 0.1638791197530417,
    "kalman.n_eff@25": 7.999488060097996,
    "kalman.n_eff@60": 12.473100285951407,
    "kalman.n_eff@119": 15.110060335371337,
    "kalman.settled_frac@25": 0.5136725262938573,
    "kalman.settled_frac@60": 0.8564127056253706,
    "kalman.settled_frac@119": 0.9782071302132749,
    "kalman.withheld_reason@25": None,
    "kalman.withheld_reason@60": None,
    "kalman.withheld_reason@119": None,
    "lasso.pred_y0__l0.2@25": -4.355626741152896,
    "lasso.pred_y0__l0.2@60": -0.17105722624329744,
    "lasso.pred_y0__l0.2@119": -0.2636883958221261,
    "lasso.resid_y0__l0.2@25": -0.4959790830818198,
    "lasso.resid_y0__l0.2@60": -0.3009839025737805,
    "lasso.resid_y0__l0.2@119": 0.21298962486389728,
    "lasso.pred_y0__l0@25": -4.68437561144585,
    "lasso.pred_y0__l0@60": -0.25563260674706156,
    "lasso.pred_y0__l0@119": -0.15976613727912523,
    "lasso.resid_y0__l0@25": -0.16723021278886652,
    "lasso.resid_y0__l0@60": -0.21640852207001637,
    "lasso.resid_y0__l0@119": 0.10906736632089639,
    "lasso.n_eff@25": 7.999488060097996,
    "lasso.n_eff@60": 12.473100285951407,
    "lasso.n_eff@119": 15.110060335371337,
    "lasso.settled_frac@25": 0.5136725262938573,
    "lasso.settled_frac@60": 0.8564127056253706,
    "lasso.settled_frac@119": 0.9782071302132749,
    "lasso.withheld_reason@25": None,
    "lasso.withheld_reason@60": None,
    "lasso.withheld_reason@119": None,
    "lasso.lam_selected_y0@25": 0.0,
    "lasso.lam_selected_y0@60": 0.0,
    "lasso.lam_selected_y0@119": 0.0,
    "huber.pred_y0@25": -4.684375830329454,
    "huber.pred_y0@60": -0.25595493316232676,
    "huber.pred_y0@119": -0.160597355524417,
    "huber.resid_y0@25": -0.16722999390526194,
    "huber.resid_y0@60": -0.21608619565475118,
    "huber.resid_y0@119": 0.10989858456618817,
    "huber.n_eff@25": 7.999488060097996,
    "huber.n_eff@60": 12.473100285951407,
    "huber.n_eff@119": 15.110060335371337,
    "huber.settled_frac@25": 0.5136725262938573,
    "huber.settled_frac@60": 0.8564127056253706,
    "huber.settled_frac@119": 0.9782071302132749,
    "huber.withheld_reason@25": None,
    "huber.withheld_reason@60": None,
    "huber.withheld_reason@119": None,
    "quantile.pred_y0@25": -4.684375830329454,
    "quantile.pred_y0@60": -0.19342194783921984,
    "quantile.pred_y0@119": -0.11291822930615636,
    "quantile.resid_y0@25": -0.16722999390526194,
    "quantile.resid_y0@60": -0.2786191809778581,
    "quantile.resid_y0@119": 0.062219458347927525,
    "quantile.n_eff@25": 7.999488060097996,
    "quantile.n_eff@60": 12.473100285951407,
    "quantile.n_eff@119": 15.110060335371337,
    "quantile.settled_frac@25": 0.5136725262938573,
    "quantile.settled_frac@60": 0.8564127056253706,
    "quantile.settled_frac@119": 0.9782071302132749,
    "quantile.withheld_reason@25": None,
    "quantile.withheld_reason@60": None,
    "quantile.withheld_reason@119": None,
    "sgd.pred_y0@25": -4.443842552576383,
    "sgd.pred_y0@60": -0.0818413274046676,
    "sgd.pred_y0@119": -0.13251990556028467,
    "sgd.resid_y0@25": -0.40776327165833326,
    "sgd.resid_y0@60": -0.39019980141241034,
    "sgd.resid_y0@119": 0.08182113460205584,
    "sgd.n_eff@25": 7.999488060097996,
    "sgd.n_eff@60": 12.473100285951407,
    "sgd.n_eff@119": 15.110060335371337,
    "sgd.settled_frac@25": 0.5136725262938573,
    "sgd.settled_frac@60": 0.8564127056253706,
    "sgd.settled_frac@119": 0.9782071302132749,
    "sgd.withheld_reason@25": None,
    "sgd.withheld_reason@60": None,
    "sgd.withheld_reason@119": None,
    "pa.pred_y0@25": -4.090089774120203,
    "pa.pred_y0@60": -0.1320239199951636,
    "pa.pred_y0@119": -0.15725972713536898,
    "pa.resid_y0@25": -0.7615160501145128,
    "pa.resid_y0@60": -0.34001720882191433,
    "pa.resid_y0@119": 0.10656095617714015,
    "pa.n_eff@25": 7.999488060097996,
    "pa.n_eff@60": 12.473100285951407,
    "pa.n_eff@119": 15.110060335371337,
    "pa.settled_frac@25": 0.5136725262938573,
    "pa.settled_frac@60": 0.8564127056253706,
    "pa.settled_frac@119": 0.9782071302132749,
    "pa.withheld_reason@25": None,
    "pa.withheld_reason@60": None,
    "pa.withheld_reason@119": None,
    "sgd_simplex.pred_y0@25": 0.6014530718057314,
    "sgd_simplex.pred_y0@60": -2.043111097655836,
    "sgd_simplex.pred_y0@119": -1.2501428978281939,
    "sgd_simplex.resid_y0@25": -5.453058896040448,
    "sgd_simplex.resid_y0@60": 1.5710699688387582,
    "sgd_simplex.resid_y0@119": 1.199444126869965,
    "sgd_simplex.n_eff@25": 7.999488060097996,
    "sgd_simplex.n_eff@60": 12.473100285951407,
    "sgd_simplex.n_eff@119": 15.110060335371337,
    "sgd_simplex.settled_frac@25": 0.5136725262938573,
    "sgd_simplex.settled_frac@60": 0.8564127056253706,
    "sgd_simplex.settled_frac@119": 0.9782071302132749,
    "sgd_simplex.withheld_reason@25": None,
    "sgd_simplex.withheld_reason@60": None,
    "sgd_simplex.withheld_reason@119": None,
    "pa_box.pred_y0@25": -3.737638427837444,
    "pa_box.pred_y0@60": -0.33497852077198886,
    "pa_box.pred_y0@119": -0.28734384132863633,
    "pa_box.resid_y0@25": -1.113967396397272,
    "pa_box.resid_y0@60": -0.13706260804508907,
    "pa_box.resid_y0@119": 0.2366450703704075,
    "pa_box.n_eff@25": 7.999488060097996,
    "pa_box.n_eff@60": 12.473100285951407,
    "pa_box.n_eff@119": 15.110060335371337,
    "pa_box.settled_frac@25": 0.5136725262938573,
    "pa_box.settled_frac@60": 0.8564127056253706,
    "pa_box.settled_frac@119": 0.9782071302132749,
    "pa_box.withheld_reason@25": None,
    "pa_box.withheld_reason@60": None,
    "pa_box.withheld_reason@119": None,
    "kalman_revert.pred_y0@25": 0.7158232394458026,
    "kalman_revert.pred_y0@60": -1.7736562223617134,
    "kalman_revert.pred_y0@119": -0.12221200148522181,
    "kalman_revert.resid_y0@25": -5.567429063680519,
    "kalman_revert.resid_y0@60": 1.3016150935446356,
    "kalman_revert.resid_y0@119": 0.07151323052699297,
    "kalman_revert.n_eff@25": 7.999488060097996,
    "kalman_revert.n_eff@60": 12.473100285951407,
    "kalman_revert.n_eff@119": 15.110060335371337,
    "kalman_revert.settled_frac@25": 0.5136725262938573,
    "kalman_revert.settled_frac@60": 0.8564127056253706,
    "kalman_revert.settled_frac@119": 0.9782071302132749,
    "kalman_revert.withheld_reason@25": None,
    "kalman_revert.withheld_reason@60": None,
    "kalman_revert.withheld_reason@119": None,
    "ftrl.pred_y0@25": -4.22640569924921,
    "ftrl.pred_y0@60": -0.4694780613966558,
    "ftrl.pred_y0@119": -0.14312415679994062,
    "ftrl.resid_y0@25": -0.6252001249855059,
    "ftrl.resid_y0@60": -0.002563067420422116,
    "ftrl.resid_y0@119": 0.09242538584171178,
    "ftrl.n_eff@25": 7.999488060097996,
    "ftrl.n_eff@60": 12.473100285951407,
    "ftrl.n_eff@119": 15.110060335371337,
    "ftrl.settled_frac@25": 0.5136725262938573,
    "ftrl.settled_frac@60": 0.8564127056253706,
    "ftrl.settled_frac@119": 0.9782071302132749,
    "ftrl.withheld_reason@25": None,
    "ftrl.withheld_reason@60": None,
    "ftrl.withheld_reason@119": None,
    "holt.pred_y0@25": -1.6482424919748164,
    "holt.pred_y0@60": 5.630591873314062,
    "holt.pred_y0@119": -1.4684387824643939,
    "holt.resid_y0@25": -3.2033633322599,
    "holt.resid_y0@60": -6.10263300213114,
    "holt.resid_y0@119": 1.4177400115061651,
    "holt.n_eff@25": 8.573837237596514,
    "holt.n_eff@60": 13.244185655943454,
    "holt.n_eff@119": 15.09397614533663,
    "holt.settled_frac@25": 0.5136725262938573,
    "holt.settled_frac@60": 0.8564127056253706,
    "holt.settled_frac@119": 0.9793826888941736,
    "holt.withheld_reason@25": None,
    "holt.withheld_reason@60": None,
    "holt.withheld_reason@119": None,
    "moments.mean_x0@25": 0.14567573584298,
    "moments.mean_x0@60": 0.5319388857885934,
    "moments.mean_x0@119": 0.07562417774137233,
    "moments.mean_x1@25": 1.3170476393181616,
    "moments.mean_x1@60": 1.503050612553659,
    "moments.mean_x1@119": 2.0556054863392945,
    "moments.var_x0@25": 1.1771092288363498,
    "moments.var_x0@60": 1.1318174907805318,
    "moments.var_x0@119": 1.0228349789270739,
    "moments.var_x1@25": 11.367059553916071,
    "moments.var_x1@60": 5.058876857932047,
    "moments.var_x1@119": 7.436220266507603,
    "moments.corr_x0_x1@25": 0.1950011675208187,
    "moments.corr_x0_x1@60": 0.31607532904190233,
    "moments.corr_x0_x1@119": 0.2439860537587861,
    "moments.n_eff@25": 7.999488060097996,
    "moments.n_eff@60": 12.473100285951407,
    "moments.n_eff@119": 15.110060335371337,
    "moments.settled_frac@25": 0.5136725262938573,
    "moments.settled_frac@60": 0.8564127056253706,
    "moments.settled_frac@119": 0.9782071302132749,
    "moments.withheld_reason@25": None,
    "moments.withheld_reason@60": None,
    "moments.withheld_reason@119": None,
    "kmeans.cluster@25": 1,
    "kmeans.cluster@60": 0,
    "kmeans.cluster@119": 0,
    "kmeans.dist@25": 1.0657683037801138,
    "kmeans.dist@60": 2.0686144858219824,
    "kmeans.dist@119": 0.23320451009648485,
    "kmeans.dist2@25": 2.9549386449346025,
    "kmeans.dist2@60": 3.1351140254811636,
    "kmeans.dist2@119": 1.9857367773902603,
    "kmeans.n_eff@25": 7.999488060097996,
    "kmeans.n_eff@60": 12.473100285951407,
    "kmeans.n_eff@119": 15.110060335371337,
    "kmeans.settled_frac@25": 0.5136725262938573,
    "kmeans.settled_frac@60": 0.8564127056253706,
    "kmeans.settled_frac@119": 0.9782071302132749,
    "kmeans.withheld_reason@25": None,
    "kmeans.withheld_reason@60": None,
    "kmeans.withheld_reason@119": None,
    "micro.cluster@25": None,
    "micro.cluster@60": 2,
    "micro.cluster@119": 5,
    "micro.dist@25": None,
    "micro.dist@60": 2.146534886929419,
    "micro.dist@119": 0.6444533017493204,
    "micro.micro@25": 5,
    "micro.micro@60": 10,
    "micro.micro@119": 7,
    "micro.outlier@25": True,
    "micro.outlier@60": True,
    "micro.outlier@119": False,
    "micro.n_clusters@25": 0,
    "micro.n_clusters@60": 1,
    "micro.n_clusters@119": 1,
    "micro.n_micro@25": 2,
    "micro.n_micro@60": 6,
    "micro.n_micro@119": 4,
    "micro.n_eff@25": 7.999488060097996,
    "micro.n_eff@60": 12.473100285951407,
    "micro.n_eff@119": 15.110060335371337,
    "micro.settled_frac@25": 0.5136725262938573,
    "micro.settled_frac@60": 0.8564127056253706,
    "micro.settled_frac@119": 0.9782071302132749,
    "micro.withheld_reason@25": None,
    "micro.withheld_reason@60": None,
    "micro.withheld_reason@119": None,
    "ew_class.class@25": "hi",
    "ew_class.class@60": "lo",
    "ew_class.class@119": "lo",
    "ew_class.p_lo@25": 0.001769258464791603,
    "ew_class.p_lo@60": 0.9998050842334101,
    "ew_class.p_lo@119": 0.9706581254697833,
    "ew_class.p_hi@25": 0.9982307415352083,
    "ew_class.p_hi@60": 0.00019491576659006558,
    "ew_class.p_hi@119": 0.029341874530216645,
    "ew_class.n_eff@25": 7.999488060097996,
    "ew_class.n_eff@60": 12.473100285951407,
    "ew_class.n_eff@119": 15.110060335371337,
    "ew_class.settled_frac@25": 0.5136725262938573,
    "ew_class.settled_frac@60": 0.8564127056253706,
    "ew_class.settled_frac@119": 0.9782071302132749,
    "ew_class.withheld_reason@25": None,
    "ew_class.withheld_reason@60": None,
    "ew_class.withheld_reason@119": None,
    "marginal.n_eff@25": 7.999488060097996,
    "marginal.n_eff@60": 12.473100285951407,
    "marginal.n_eff@119": 15.110060335371337,
    "marginal.n_eff[a/x0]@end": 14.643505957885951,
    "marginal.n_kish[a/x0]@end": 22.588085651395215,
    "marginal.corr[a/x0]@end": 0.38573773334037753,
    "marginal.beta[a/x0]@end": 1.2316780579628355,
    "marginal.t[a/x0]@end": 1.89706698898827,
    "marginal.n_eff[a/x1]@end": 14.643505957885951,
    "marginal.n_kish[a/x1]@end": 22.588085651395215,
    "marginal.corr[a/x1]@end": -0.8869994444789747,
    "marginal.beta[a/x1]@end": -0.7081876904377251,
    "marginal.t[a/x1]@end": -8.715757847257636,
    "marginal.n_eff[b/x0]@end": 14.25784941881905,
    "marginal.n_kish[b/x0]@end": 23.20892217040503,
    "marginal.corr[b/x0]@end": 0.41913584410223514,
    "marginal.beta[b/x0]@end": 0.9210410451370622,
    "marginal.t[b/x0]@end": 2.1260076771757284,
    "marginal.n_eff[b/x1]@end": 14.25784941881905,
    "marginal.n_kish[b/x1]@end": 23.20892217040503,
    "marginal.corr[b/x1]@end": -0.7634819469727882,
    "marginal.beta[b/x1]@end": -0.6068290466325643,
    "marginal.t[b/x1]@end": -5.44427952250404,
    "corrchange.stat@25": None,
    "corrchange.stat@60": None,
    "corrchange.stat@119": None,
    "corrchange.crit@25": None,
    "corrchange.crit@60": None,
    "corrchange.crit@119": None,
    "corrchange.flag@25": None,
    "corrchange.flag@60": None,
    "corrchange.flag@119": None,
    "corrchange.since_flag@25": None,
    "corrchange.since_flag@60": None,
    "corrchange.since_flag@119": None,
    "corrchange.n_eff@25": 11.0,
    "corrchange.n_eff@60": 29.0,
    "corrchange.n_eff@119": 57.0,
    "corrchange.settled_frac@25": None,
    "corrchange.settled_frac@60": None,
    "corrchange.settled_frac@119": None,
    "corrchange.withheld_reason@25": None,
    "corrchange.withheld_reason@60": None,
    "corrchange.withheld_reason@119": None,
    "corrchange.stat#0": 1.0721348056303184,
    "corrchange.stat#1": 1.0584184395238514,
    "corrchange.stat#2": 0.9351588646528833,
    "corrchange.stat#3": 0.761834453284964,
    "hmm.p_0@25": 0.0019837560822976905,
    "hmm.p_0@60": 0.7065450485379252,
    "hmm.p_0@119": 3.6843316145567297e-07,
    "hmm.p_1@25": 0.9980162439177024,
    "hmm.p_1@60": 0.29345495146207473,
    "hmm.p_1@119": 0.9999996315668386,
    "hmm.p1_0@25": 0.29240113431229137,
    "hmm.p1_0@60": 0.5116753621005758,
    "hmm.p1_0@119": 0.2119510684647355,
    "hmm.p1_1@25": 0.7075988656877088,
    "hmm.p1_1@60": 0.48832463789942415,
    "hmm.p1_1@119": 0.7880489315352647,
    "hmm.state@25": 1,
    "hmm.state@60": 0,
    "hmm.state@119": 1,
    "hmm.loglik@25": -4.532665819206047,
    "hmm.loglik@60": -5.885076725731101,
    "hmm.loglik@119": -3.086778694060611,
    "hmm.n_eff@25": 7.999488060097996,
    "hmm.n_eff@60": 12.473100285951407,
    "hmm.n_eff@119": 15.110060335371337,
    "hmm.settled_frac@25": 0.5136725262938573,
    "hmm.settled_frac@60": 0.8564127056253706,
    "hmm.settled_frac@119": 0.9782071302132749,
    "hmm.withheld_reason@25": None,
    "hmm.withheld_reason@60": None,
    "hmm.withheld_reason@119": None,
    "rcov.n_eff@25": 11.0,
    "rcov.n_eff@60": 0.0,
    "rcov.n_eff@119": 28.0,
    "deco.u@25": 0.4167923065360934,
    "deco.u@60": 0.9917268375727508,
    "deco.u@119": 0.8126243515712434,
    "deco.rho@25": 0.576990565121172,
    "deco.rho@60": -0.10467461411897723,
    "deco.rho@119": 0.12882376266398168,
    "deco.loglik@25": -3.9973745744499514,
    "deco.loglik@60": -6.086831238680253,
    "deco.loglik@119": -2.2061901908973724,
    "deco.n_eff@25": 7.999488060097996,
    "deco.n_eff@60": 12.473100285951407,
    "deco.n_eff@119": 15.110060335371337,
    "deco.settled_frac@25": 0.5136725262938573,
    "deco.settled_frac@60": 0.8564127056253706,
    "deco.settled_frac@119": 0.9782071302132749,
    "deco.withheld_reason@25": None,
    "deco.withheld_reason@60": None,
    "deco.withheld_reason@119": None,
    "bocpd.p_change@25": 0.0252003885763477,
    "bocpd.p_change@60": 0.062157054707882006,
    "bocpd.p_change@119": 0.15789808629898666,
    "bocpd.run_mode@25": 11,
    "bocpd.run_mode@60": 29,
    "bocpd.run_mode@119": 49,
    "bocpd.run_mean@25": 10.70353313482607,
    "bocpd.run_mean@60": 11.563346786776185,
    "bocpd.run_mean@119": 47.093702164952965,
    "bocpd.pred_x0@25": 0.10193127582805904,
    "bocpd.pred_x0@60": 0.19336627428835312,
    "bocpd.pred_x0@119": -0.08401090273579666,
    "bocpd.pred_x1@25": 1.0700197314046975,
    "bocpd.pred_x1@60": 0.607491296609566,
    "bocpd.pred_x1@119": 2.375653796829278,
    "bocpd.logscore@25": -5.660395809868995,
    "bocpd.logscore@60": -7.568874535614931,
    "bocpd.logscore@119": -3.1370975353029267,
    "bocpd.n_eff@25": 11.0,
    "bocpd.n_eff@60": 28.5,
    "bocpd.n_eff@119": 57.0,
    "bocpd.settled_frac@25": None,
    "bocpd.settled_frac@60": None,
    "bocpd.settled_frac@119": None,
    "bocpd.withheld_reason@25": None,
    "bocpd.withheld_reason@60": None,
    "bocpd.withheld_reason@119": None,
    "seqtest.log_e_pos_y0@25": 0.0,
    "seqtest.log_e_pos_y0@60": 0.0,
    "seqtest.log_e_pos_y0@119": 0.0,
    "seqtest.log_e_neg_y0@25": 2.797424236204631,
    "seqtest.log_e_neg_y0@60": -1.7626699454934798,
    "seqtest.log_e_neg_y0@119": 5.481151387104117,
    "seqtest.n_pos_y0@25": 1,
    "seqtest.n_pos_y0@60": 13,
    "seqtest.n_pos_y0@119": 14,
    "seqtest.n_neg_y0@25": 10,
    "seqtest.n_neg_y0@60": 16,
    "seqtest.n_neg_y0@119": 43,
    "seqtest.n_eff@25": 12.0,
    "seqtest.n_eff@60": 30.0,
    "seqtest.n_eff@119": 59.0,
    "seqtest.settled_frac@25": None,
    "seqtest.settled_frac@60": None,
    "seqtest.settled_frac@119": None,
    "seqtest.withheld_reason@25": None,
    "seqtest.withheld_reason@60": None,
    "seqtest.withheld_reason@119": None,
    "seqtest_compare.log_e_a_y0@25": 0.0,
    "seqtest_compare.log_e_a_y0@60": 8.308417683424773,
    "seqtest_compare.log_e_a_y0@119": 0.24797582013043007,
    "seqtest_compare.log_e_b_y0@25": -0.9808292530117262,
    "seqtest_compare.log_e_b_y0@60": -0.9808292530117262,
    "seqtest_compare.log_e_b_y0@119": -0.9808292530117262,
    "seqtest_compare.wins_a_y0@25": 2,
    "seqtest_compare.wins_a_y0@60": 21,
    "seqtest_compare.wins_a_y0@119": 31,
    "seqtest_compare.wins_b_y0@25": 2,
    "seqtest_compare.wins_b_y0@60": 2,
    "seqtest_compare.wins_b_y0@119": 19,
    "seqtest_compare.n_eff@25": 12.0,
    "seqtest_compare.n_eff@60": 30.0,
    "seqtest_compare.n_eff@119": 59.0,
    "seqtest_compare.settled_frac@25": None,
    "seqtest_compare.settled_frac@60": None,
    "seqtest_compare.settled_frac@119": None,
    "seqtest_compare.withheld_reason@25": None,
    "seqtest_compare.withheld_reason@60": None,
    "seqtest_compare.withheld_reason@119": None,
    "rcov.rcov_n[rcov/a/m]": 21,
    "rcov.bandwidth[rcov/a/m]": 3,
    "rcov.rcov0[rcov/a/m]": 21.95819492943922,
    "rcov.rcov1[rcov/a/m]": 62.2773177861597,
    "rcov.rcov2[rcov/a/m]": 353.6268031231857,
    "rcov.rcov_n[rcov/b/m]": 21,
    "rcov.bandwidth[rcov/b/m]": 3,
    "rcov.rcov0[rcov/b/m]": 20.168462820112076,
    "rcov.rcov1[rcov/b/m]": 17.798995434502125,
    "rcov.rcov2[rcov/b/m]": 762.0001389768797,
}


# An emptied table fails here as a changed schema: a skip would have turned
# the cross-platform check off without a word.
def test_the_pipeline_produces_the_same_numbers_everywhere():
    got = signature()
    assert set(got) == set(GOLDEN), (
        "the output schema changed: "
        f"added {sorted(set(got) - set(GOLDEN))}, removed {sorted(set(GOLDEN) - set(got))}"
    )
    for key, want in GOLDEN.items():
        have = got[key]
        if want is None or have is None:
            assert have == want, f"{key}: {have} vs {want} (null-ness must match)"
            continue
        if isinstance(want, str | bool):
            # A class name or a flag: exact, or it is a different answer.
            assert have == want, f"{key}: {have!r} vs {want!r}"
            continue
        assert abs(have - want) <= TOL * (1.0 + abs(want)), (
            f"{key}: {have!r} vs {want!r} (relative {abs(have - want) / (1 + abs(want)):.2e})"
        )


def test_the_stream_exercises_what_it_claims_to():
    """A guard on the fixture: if it stopped containing nulls or a clock gap,
    the golden comparison would still pass while covering less."""
    df = stream()
    assert df["x1"].null_count() > 0, "no null feature"
    assert df["y0"].null_count() > 0, "no null target"
    assert df["label"].null_count() > 0, "no null label"
    assert df["label"].n_unique() == 3, "the label must hold both classes and null"
    assert df["g"].n_unique() == 2, "not two groups"
    assert df["session"].n_unique() == 2, "no session break"
    assert df["w"].n_unique() > 1, "weights are constant"
    gaps = df["t"].diff().drop_nulls()
    assert gaps.max() > 6.0, "no gap beyond max_dclock"
    assert gaps.min() > 0.0, "the clock must be strictly increasing"


if __name__ == "__main__":
    # Regeneration, run as a script rather than a test that skips unless
    # asked: prints the table to paste over `GOLDEN` above.
    print("GOLDEN = {")
    for key, value in signature().items():
        print(f"    {key!r}: {value!r},")
    print("}")
