"""T-W7: fixed numbers out of the whole pipeline, compared on every OS.

`crates/online-core/tests/golden.rs` pins the Rust core's arithmetic, but
nothing pinned what comes out of the *Polars* layer -- extraction, the
per-group fan-out, the diagnostics, the struct assembly. A divergence
introduced there, or by polars' own vectorized paths on a different CPU, would
be invisible until someone compared two machines by hand.

This is that comparison, made automatic: the numbers are committed, and CI runs
this file on ubuntu, macOS and Windows. Locally the agreement is exact; the
tolerance is what "the same answer on another platform" is allowed to mean.

The constants come from the current implementation, which is held
elsewhere to an oracle it cannot share a bug with: the ten regression models
to the numpy references in `tests/reference.py` and `tests/reference_paths.py`,
and the other kinds to the oracles `docs/TESTING.md`'s table lists (their
definitions, papers and second opinions) -- the same bargain `golden.rs`
makes. Regenerate only after confirming a change is intended:

    uv run python tests/test_golden_pipeline.py
"""

from datetime import datetime, timedelta

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
        gap_cap=6.0,
        half_life=25.0,
        weight="w",
        group="g",
        min_weight=4.0,
    )
    return [
        po.spec.ewridge(
            "ridge",
            ridge=[1e-6, 0.5],
            standardize=True,
            max_rows_between_solves=1,
            emit_sigma=True,
            emit_zscore=True,
            conformal=0.9,
            **common,
        ),
        po.spec.rls("rls", delta=1.0, **common),
        po.spec.kalman("kalman", coef_half_life=80.0, **common),
        po.spec.lasso("lasso", lasso_path=[0.2, 0.0], max_rows_between_solves=1, **common),
        po.spec.huber("huber", max_rows_between_solves=1, **common),
        po.spec.quantile("quantile", quantile=0.75, max_rows_between_solves=1, **common),
        po.spec.sgd("sgd", learning_rate=0.02, **common),
        po.spec.pa("pa", **common),
        po.spec.sgd("sgd_simplex", learning_rate=0.02, coef_min=0.0, coef_sum=1.0, **common),
        po.spec.pa("pa_box", coef_min=-0.5, coef_max=0.5, **common),
        po.spec.kalman(
            "kalman_revert", coef_half_life=80.0, revert_half_life=[INF, 30.0, 8.0], **common
        ),
        po.spec.ftrl("ftrl", loss="squared", alpha=0.5, **common),
        po.spec.holt(
            "holt",
            targets=["y0"],
            clock="t",
            gap_cap=6.0,
            half_life=25.0,
            weight="w",
            group="g",
            min_weight=4.0,
        ),
        po.spec.ew_cov(
            "moments",
            features=["x0", "x1"],
            stats=["mean", "var", "corr"],
            clock="t",
            gap_cap=6.0,
            half_life=25.0,
            weight="w",
            group="g",
            min_weight=4.0,
        ),
        po.spec.kmeans(
            "kmeans",
            features=["x0", "x1"],
            k=2,
            warm_rows=8,
            split_merge=0.5,
            split_merge_every_rows=10,
            clock="t",
            gap_cap=6.0,
            half_life=25.0,
            weight="w",
            group="g",
            min_weight=4.0,
        ),
        po.spec.micro(
            "micro",
            features=["x0", "x1"],
            eps=0.5,
            beta_mu=2.0,
            prune_every=10,
            clock="t",
            gap_cap=6.0,
            half_life=25.0,
            weight="w",
            group="g",
            min_weight=4.0,
        ),
        po.spec.ew_class(
            "ew_class",
            features=["x0", "x1"],
            label="label",
            classes=["lo", "hi"],
            covariance="shared",
            precision_prior=0.5,
            clock="t",
            gap_cap=6.0,
            half_life=25.0,
            weight="w",
            group="g",
            min_weight=4.0,
        ),
        # Only `weight_sum` per row; the pairs are read from the state at the end.
        po.spec.marginal("marginal", **common),
        # A constancy test over spans of 30 rows, so the 120-row stream
        # closes several per group.
        # A cap above the stream's 9-unit gaps: at `gap_cap = 6` every
        # gap is capped, `clear_lags` abandons the span (task 47), and no
        # span ever closes -- which is correct and pins nothing.
        po.spec.corrchange(
            "corrchange",
            features=["x0", "x1"],
            span_rows=20,
            alpha=0.05,
            clock="t",
            gap_cap=100.0,
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
            gap_cap=6.0,
            half_life=25.0,
            weight="w",
            group="g",
        ),
        # A block's realised covariance. The stream's groups interleave, so
        # the close is on the session, not monotone; the block itself is
        # pinned below, since nothing but `weight_sum` is emitted per row.
        po.spec.rcov(
            "rcov",
            features=["x0", "x1"],
            kind="kernel",
            bandwidth=3,
            clock="t",
            gap_cap=6.0,
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
            gap_cap=6.0,
            half_life=25.0,
            weight="w",
            group="g",
            min_weight=4.0,
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
            gap_cap=6.0,
            weight="w",
            group="g",
            min_weight=4.0,
        ),
        # No weight and no half-life: a test counts trials and does not forget.
        po.spec.seqtest(
            "seqtest", targets=["y0"], clock="t", gap_cap=6.0, group="g", min_weight=4.0
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
            gap_cap=6.0,
            group="g",
            min_weight=4.0,
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
                for field in ("weight_sum", "n_kish", "corr", "beta", "t_stat"):
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
        sig[f"{row['spec']}.bandwidth{where}"] = row["rcov_bandwidth_used"]
        for i, v in enumerate(row["rcov"] or []):
            sig[f"{row['spec']}.rcov{i}{where}"] = v
    return sig


def switched_stream(n: int = 120) -> pl.DataFrame:
    """`stream()` with what the second bank reads beside it: the clock as a
    `Datetime`, a minute for each unit of ``t``; a weight column with a row
    of weight 0 every seventh row; and a drifting level ``mid`` for the
    formula target."""
    df = stream(n)
    start = datetime(2024, 1, 2, 9, 30)
    ts = [start + timedelta(minutes=float(v)) for v in df["t"].to_list()]
    wz = [0.0 if i % 7 == 3 else v for i, v in enumerate(df["w"].to_list())]
    mid = 100.0 + np.cumsum(0.1 * df["x0"].to_numpy())
    return df.with_columns(
        pl.Series("ts", ts, dtype=pl.Datetime("us")), pl.Series("wz", wz), pl.Series("mid", mid)
    )


def switched_specs() -> list[dict]:
    """One spec per switch the first bank never turns (review 2026-10-06,
    TA12), each on the paths most likely to part by platform: a window, an
    embargo, a temporal clock read in integer nanoseconds with its
    parameters as durations, ``session_shrink``'s blend, pairwise gaps, no
    intercept, a coefficient prior, feature sets, every residual diagnostic,
    a formula target, rows of weight 0 (every spec reads ``wz``), and
    `bocpd`'s and `micro`'s clock parameters as durations. Clear of what task
    186 moves: no target that starts late in a solving model, no windowed
    `ew_class`, no `share_p`, no `rcov`."""
    common = dict(
        targets=["y0"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=6.0,
        half_life=25.0,
        weight="wz",
        group="g",
        min_weight=4.0,
    )
    timed = dict(common, clock="ts", gap_cap="6m", half_life="25m")
    fwd = (po.rewm_mean("mid", half_life=5.0, window_size=10.0) - pl.col("mid")).alias("fwd")
    unsupervised = {k: v for k, v in timed.items() if k not in ("targets", "min_weight")}
    return [
        # Under `closed="both"`, the edge every window had before task 196,
        # so its numbers are the ones pinned; the default `"right"` drops a
        # row exactly 20 old, moving `pred_y0@60` by 7.3e-5, and is held to
        # the definition in `test_window.py` and `test_second_opinion.py`.
        po.spec.ewridge(
            "window", window_size=20.0, closed="both", max_rows_between_solves=1, **common
        ),
        po.spec.kalman("embargo", coef_half_life=80.0, embargo=3.0, **common),
        po.spec.ewridge("datetime", max_rows_between_solves=1, **timed),
        po.spec.ewridge(
            "session_shrink",
            session="session",
            session_gap=3.0,
            session_shrink=0.5,
            long_half_life=100.0,
            max_rows_between_solves=1,
            **common,
        ),
        po.spec.ewridge("pairwise", target_gaps="pairwise", max_rows_between_solves=1, **common),
        po.spec.ewridge("origin", fit_intercept=False, max_rows_between_solves=1, **common),
        po.spec.rls("prior", delta=1.0, coef_prior=[[0.2, 1.0, -0.5]], **common),
        po.spec.ewridge(
            "feature_sets",
            feature_sets={"one": ["x0"], "both": ["x0", "x1"]},
            max_rows_between_solves=1,
            **common,
        ),
        po.spec.ewridge(
            "diagnostics",
            ridge=[1e-6, 0.5],
            max_rows_between_solves=1,
            emit_sigma=True,
            emit_zscore=True,
            emit_drift=True,
            drift_threshold=20.0,
            resid_quantiles=[0.5, 0.9],
            emit_autocorr=True,
            emit_metrics=True,
            emit_selected=True,
            emit_averaged=True,
            emit_error_inflation=True,
            emit_clocks=True,
            **common,
        ),
        # A target resolved once its window has closed, so scored from early
        # on (no `min_weight`) to pin rows 25 and 60 as well as 119.
        po.spec.kalman(
            "formula",
            coef_half_life=80.0,
            embargo=12.0,
            **{**common, "targets": [fwd], "min_weight": 0.0},
        ),
        po.spec.bocpd(
            "hazard",
            hazard="40m",
            emission="diag",
            prior_scale=[1.0],
            max_run=50,
            **{k: v for k, v in unsupervised.items() if k != "half_life"},
        ),
        po.spec.micro("prune", eps=0.5, beta_mu=2.0, prune_every="10m", **unsupervised),
    ]


def switched_signature() -> dict[str, float | str | None]:
    """Every non-coefficient output field of the second bank, at the same
    three rows."""
    bank = po.ModelBank(switched_specs())
    out = bank.fit_predict(switched_stream())
    sig: dict[str, float | str | None] = {}
    for spec in switched_specs():
        name = spec["name"]
        for field in po.spec.output_fields(spec):
            if field.startswith(("coef", "support_coef")):
                continue
            values = out[name].struct.field(field).to_list()
            for row in PICKS:
                value = values[row]
                # A `Datetime` clock field, pinned as its text.
                sig[f"{name}.{field}@{row}"] = str(value) if isinstance(value, datetime) else value
    return sig


#: Produced by `uv run python tests/test_golden_pipeline.py`. Re-pinned for
#: docs/PLAN.md task 195 (2026-10-07), whose defaults moved 45 of these and
#: 16 of `SWITCHED_GOLDEN`'s: `huber`'s `huber_delta` 1.345 (U3), `sgd`'s and
#: `pa`'s `standardize` on (U2) and `pa`'s tube in residual stds (U1), and
#: `bocpd`'s prior from the first rows with `nu` per emission (U4, U5). With
#: the old defaults given -- `huber_delta = 1.5`, `standardize = False`, a
#: zero mean, the identity and `nu = d + 2` -- every one of them but `pa`'s
#: and `pa_box`'s comes back to the old value within `TOL`; `pa`'s are a
#: replica of the standardized step with the sigma tube, written from the
#: docstrings, to 5.4e-16 on all 104 rows it predicts. Task 202 (the same
#: day) moved those 12 again, `pa`'s and `pa_box`'s `pred` and `resid`: the
#: tube is in units of the target's own EW std. A replica written from the
#: docstrings, the standardized PA-I step with that tube (its variance from
#: the definition) and, for `pa_box`, the box clamped in standardized
#: coordinates, gives `pa`'s to 1.2e-15 and `pa_box`'s to 5.6e-16 on all 104
#: rows each predicts; nothing else moved. Task 203 (2026-10-07) moved the
#: same 12 once more: `pa`'s `eps` defaults to 0.01, not 0.1. That replica at
#: 0.01 gives `pa`'s to 1.3e-15 and `pa_box`'s to 4.4e-16 on all 104 rows,
#: with the same nulls, and at 0.1 gives back the old 12 to 3.4e-15; nothing
#: else moved.
GOLDEN: dict[str, float | str | None] = {
    "ridge.pred_y0__r0.000001@25": -4.684371132566456,
    "ridge.pred_y0__r0.000001@60": -0.25563207972202284,
    "ridge.pred_y0__r0.000001@119": -0.15976731834715135,
    "ridge.resid_y0__r0.000001@25": -0.1672346916682601,
    "ridge.resid_y0__r0.000001@60": -0.2164090490950551,
    "ridge.resid_y0__r0.000001@119": 0.10906854738892252,
    "ridge.pred_y0__r0.5@25": -3.2074053272438663,
    "ridge.pred_y0__r0.5@60": -0.1169734771432751,
    "ridge.pred_y0__r0.5@119": -0.5303952399350176,
    "ridge.resid_y0__r0.5@25": -1.64420049699085,
    "ridge.resid_y0__r0.5@60": -0.35506765167380283,
    "ridge.resid_y0__r0.5@119": 0.47969646897678875,
    "ridge.sigma_y0__r0.000001@25": 0.051493183224268005,
    "ridge.sigma_y0__r0.000001@60": 0.14974398485680734,
    "ridge.sigma_y0__r0.000001@119": 0.09333817729646239,
    "ridge.sigma_y0__r0.5@25": 3.1349106139192906,
    "ridge.sigma_y0__r0.5@60": 0.6856591658858987,
    "ridge.sigma_y0__r0.5@119": 0.854495805542607,
    "ridge.zscore_y0__r0.000001@25": -3.247705447532806,
    "ridge.zscore_y0__r0.000001@60": -1.4451936036161734,
    "ridge.zscore_y0__r0.000001@119": 1.168530932873234,
    "ridge.zscore_y0__r0.5@25": -0.5244808224165783,
    "ridge.zscore_y0__r0.5@60": -0.5178486182927948,
    "ridge.zscore_y0__r0.5@119": 0.5613795478752295,
    "ridge.lo_y0__r0.000001@25": -4.7831691229254405,
    "ridge.lo_y0__r0.000001@60": -0.42565243395240016,
    "ridge.lo_y0__r0.000001@119": -0.2911458059377575,
    "ridge.lo_y0__r0.5@25": -12.12008108757161,
    "ridge.lo_y0__r0.5@60": -0.987459316113378,
    "ridge.lo_y0__r0.5@119": -9.116001200187965,
    "ridge.hi_y0__r0.000001@25": -4.585573142207472,
    "ridge.hi_y0__r0.000001@60": -0.0856117254916455,
    "ridge.hi_y0__r0.000001@119": -0.028388830756545164,
    "ridge.hi_y0__r0.5@25": 5.705270433083878,
    "ridge.hi_y0__r0.5@60": 0.7535123618268278,
    "ridge.hi_y0__r0.5@119": 8.055210720317929,
    "ridge.coverage_y0__r0.000001@25": 1.0,
    "ridge.coverage_y0__r0.000001@60": 0.708544385155888,
    "ridge.coverage_y0__r0.000001@119": 0.746039777195224,
    "ridge.coverage_y0__r0.5@25": 1.0,
    "ridge.coverage_y0__r0.5@60": 0.8207252289588411,
    "ridge.coverage_y0__r0.5@119": 1.0,
    "ridge.weight_sum@25": 7.999488060097996,
    "ridge.weight_sum@60": 12.473100285951407,
    "ridge.weight_sum@119": 15.110060335371337,
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
    "rls.weight_sum@25": 7.999488060097996,
    "rls.weight_sum@60": 12.473100285951407,
    "rls.weight_sum@119": 15.110060335371337,
    "rls.settled_frac@25": 0.5136725262938573,
    "rls.settled_frac@60": 0.8564127056253706,
    "rls.settled_frac@119": 0.9782071302132749,
    "rls.withheld_reason@25": None,
    "rls.withheld_reason@60": None,
    "rls.withheld_reason@119": None,
    "kalman.pred_y0@25": -0.6299441602651041,
    "kalman.pred_y0@60": -0.5177484630617406,
    "kalman.pred_y0@119": -0.22833841135939936,
    "kalman.resid_y0@25": -4.221661663969612,
    "kalman.resid_y0@60": 0.04570733424466267,
    "kalman.resid_y0@119": 0.17763964040117053,
    "kalman.weight_sum@25": 7.999488060097996,
    "kalman.weight_sum@60": 12.473100285951407,
    "kalman.weight_sum@119": 15.110060335371337,
    "kalman.settled_frac@25": 0.5136725262938573,
    "kalman.settled_frac@60": 0.8564127056253706,
    "kalman.settled_frac@119": 0.9782071302132749,
    "kalman.withheld_reason@25": None,
    "kalman.withheld_reason@60": None,
    "kalman.withheld_reason@119": None,
    "lasso.pred_y0__l0.2@25": -4.355626741152896,
    "lasso.pred_y0__l0.2@60": -0.1710572262432979,
    "lasso.pred_y0__l0.2@119": -0.26368839582212644,
    "lasso.resid_y0__l0.2@25": -0.4959790830818198,
    "lasso.resid_y0__l0.2@60": -0.30098390257378005,
    "lasso.resid_y0__l0.2@119": 0.2129896248638976,
    "lasso.pred_y0__l0@25": -4.68437561144585,
    "lasso.pred_y0__l0@60": -0.255632606747062,
    "lasso.pred_y0__l0@119": -0.15976613727912542,
    "lasso.resid_y0__l0@25": -0.16723021278886652,
    "lasso.resid_y0__l0@60": -0.21640852207001593,
    "lasso.resid_y0__l0@119": 0.10906736632089659,
    "lasso.weight_sum@25": 7.999488060097996,
    "lasso.weight_sum@60": 12.473100285951407,
    "lasso.weight_sum@119": 15.110060335371337,
    "lasso.settled_frac@25": 0.5136725262938573,
    "lasso.settled_frac@60": 0.8564127056253706,
    "lasso.settled_frac@119": 0.9782071302132749,
    "lasso.withheld_reason@25": None,
    "lasso.withheld_reason@60": None,
    "lasso.withheld_reason@119": None,
    "lasso.penalty_selected_y0@25": 0.0,
    "lasso.penalty_selected_y0@60": 0.0,
    "lasso.penalty_selected_y0@119": 0.0,
    "huber.pred_y0@25": -4.684375830329454,
    "huber.pred_y0@60": -0.25626086249461966,
    "huber.pred_y0@119": -0.1610180929365141,
    "huber.resid_y0@25": -0.16722999390526194,
    "huber.resid_y0@60": -0.21578026632245828,
    "huber.resid_y0@119": 0.11031932197828526,
    "huber.weight_sum@25": 7.999488060097996,
    "huber.weight_sum@60": 12.473100285951407,
    "huber.weight_sum@119": 15.110060335371337,
    "huber.settled_frac@25": 0.5136725262938573,
    "huber.settled_frac@60": 0.8564127056253706,
    "huber.settled_frac@119": 0.9782071302132749,
    "huber.withheld_reason@25": None,
    "huber.withheld_reason@60": None,
    "huber.withheld_reason@119": None,
    "quantile.pred_y0@25": -4.684375830329454,
    "quantile.pred_y0@60": -0.23984991107486686,
    "quantile.pred_y0@119": -0.112730473020879,
    "quantile.resid_y0@25": -0.16722999390526194,
    "quantile.resid_y0@60": -0.23219121774221108,
    "quantile.resid_y0@119": 0.06203170206265017,
    "quantile.weight_sum@25": 7.999488060097996,
    "quantile.weight_sum@60": 12.473100285951407,
    "quantile.weight_sum@119": 15.110060335371337,
    "quantile.settled_frac@25": 0.5136725262938573,
    "quantile.settled_frac@60": 0.8564127056253706,
    "quantile.settled_frac@119": 0.9782071302132749,
    "quantile.withheld_reason@25": None,
    "quantile.withheld_reason@60": None,
    "quantile.withheld_reason@119": None,
    "sgd.pred_y0@25": -0.5042876466916486,
    "sgd.pred_y0@60": -0.15162153453453298,
    "sgd.pred_y0@119": -0.3442655537531709,
    "sgd.resid_y0@25": -4.347318177543068,
    "sgd.resid_y0@60": -0.32041959428254496,
    "sgd.resid_y0@119": 0.2935667827949421,
    "sgd.weight_sum@25": 7.999488060097996,
    "sgd.weight_sum@60": 12.473100285951407,
    "sgd.weight_sum@119": 15.110060335371337,
    "sgd.settled_frac@25": 0.5136725262938573,
    "sgd.settled_frac@60": 0.8564127056253706,
    "sgd.settled_frac@119": 0.9782071302132749,
    "sgd.withheld_reason@25": None,
    "sgd.withheld_reason@60": None,
    "sgd.withheld_reason@119": None,
    "pa.pred_y0@25": -3.2785908646795354,
    "pa.pred_y0@60": 0.013829714062611576,
    "pa.pred_y0@119": -0.2540070442584794,
    "pa.resid_y0@25": -1.5730149595551808,
    "pa.resid_y0@60": -0.4858708428796895,
    "pa.resid_y0@119": 0.20330827330025059,
    "pa.weight_sum@25": 7.999488060097996,
    "pa.weight_sum@60": 12.473100285951407,
    "pa.weight_sum@119": 15.110060335371337,
    "pa.settled_frac@25": 0.5136725262938573,
    "pa.settled_frac@60": 0.8564127056253706,
    "pa.settled_frac@119": 0.9782071302132749,
    "pa.withheld_reason@25": None,
    "pa.withheld_reason@60": None,
    "pa.withheld_reason@119": None,
    "sgd_simplex.pred_y0@25": 0.6901893652038122,
    "sgd_simplex.pred_y0@60": -1.9767264663866784,
    "sgd_simplex.pred_y0@119": -1.3784973797611744,
    "sgd_simplex.resid_y0@25": -5.5417951894385284,
    "sgd_simplex.resid_y0@60": 1.5046853375696005,
    "sgd_simplex.resid_y0@119": 1.3277986088029454,
    "sgd_simplex.weight_sum@25": 7.999488060097996,
    "sgd_simplex.weight_sum@60": 12.473100285951407,
    "sgd_simplex.weight_sum@119": 15.110060335371337,
    "sgd_simplex.settled_frac@25": 0.5136725262938573,
    "sgd_simplex.settled_frac@60": 0.8564127056253706,
    "sgd_simplex.settled_frac@119": 0.9782071302132749,
    "sgd_simplex.withheld_reason@25": None,
    "sgd_simplex.withheld_reason@60": None,
    "sgd_simplex.withheld_reason@119": None,
    "pa_box.pred_y0@25": -2.9977839049458024,
    "pa_box.pred_y0@60": 0.82600431177703,
    "pa_box.pred_y0@119": -0.2291221775538278,
    "pa_box.resid_y0@25": -1.8538219192889138,
    "pa_box.resid_y0@60": -1.298045440594108,
    "pa_box.resid_y0@119": 0.17842340659559897,
    "pa_box.weight_sum@25": 7.999488060097996,
    "pa_box.weight_sum@60": 12.473100285951407,
    "pa_box.weight_sum@119": 15.110060335371337,
    "pa_box.settled_frac@25": 0.5136725262938573,
    "pa_box.settled_frac@60": 0.8564127056253706,
    "pa_box.settled_frac@119": 0.9782071302132749,
    "pa_box.withheld_reason@25": None,
    "pa_box.withheld_reason@60": None,
    "pa_box.withheld_reason@119": None,
    "kalman_revert.pred_y0@25": -0.02009119013134438,
    "kalman_revert.pred_y0@60": -1.3415903889730219,
    "kalman_revert.pred_y0@119": -0.7283624639508658,
    "kalman_revert.resid_y0@25": -4.831514634103372,
    "kalman_revert.resid_y0@60": 0.8695492601559439,
    "kalman_revert.resid_y0@119": 0.677663692992637,
    "kalman_revert.weight_sum@25": 7.999488060097996,
    "kalman_revert.weight_sum@60": 12.473100285951407,
    "kalman_revert.weight_sum@119": 15.110060335371337,
    "kalman_revert.settled_frac@25": 0.5136725262938573,
    "kalman_revert.settled_frac@60": 0.8564127056253706,
    "kalman_revert.settled_frac@119": 0.9782071302132749,
    "kalman_revert.withheld_reason@25": None,
    "kalman_revert.withheld_reason@60": None,
    "kalman_revert.withheld_reason@119": None,
    "ftrl.pred_y0@25": -4.230561129145586,
    "ftrl.pred_y0@60": -0.4392429331569383,
    "ftrl.pred_y0@119": -0.14841326593234577,
    "ftrl.resid_y0@25": -0.6210446950891306,
    "ftrl.resid_y0@60": -0.03279819566013964,
    "ftrl.resid_y0@119": 0.09771449497411694,
    "ftrl.weight_sum@25": 7.999488060097996,
    "ftrl.weight_sum@60": 12.473100285951407,
    "ftrl.weight_sum@119": 15.110060335371337,
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
    "holt.weight_sum@25": 8.573837237596514,
    "holt.weight_sum@60": 13.244185655943454,
    "holt.weight_sum@119": 15.09397614533663,
    "holt.settled_frac@25": 0.5136725262938573,
    "holt.settled_frac@60": 0.8564127056253706,
    "holt.settled_frac@119": 0.9793826888941736,
    "holt.withheld_reason@25": None,
    "holt.withheld_reason@60": None,
    "holt.withheld_reason@119": None,
    "moments.mean_x0@25": 0.14567573584298,
    "moments.mean_x0@60": 0.5319388857885933,
    "moments.mean_x0@119": 0.07562417774137233,
    "moments.mean_x1@25": 1.3170476393181616,
    "moments.mean_x1@60": 1.503050612553659,
    "moments.mean_x1@119": 2.055605486339294,
    "moments.var_x0@25": 1.1771092288363498,
    "moments.var_x0@60": 1.1318174907805318,
    "moments.var_x0@119": 1.0228349789270739,
    "moments.var_x1@25": 11.367059553916071,
    "moments.var_x1@60": 5.058876857932047,
    "moments.var_x1@119": 7.436220266507603,
    "moments.corr_x0_x1@25": 0.1950011675208187,
    "moments.corr_x0_x1@60": 0.3160753290419023,
    "moments.corr_x0_x1@119": 0.2439860537587861,
    "moments.weight_sum@25": 7.999488060097996,
    "moments.weight_sum@60": 12.473100285951407,
    "moments.weight_sum@119": 15.110060335371337,
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
    "kmeans.dist_second@25": 2.9549386449346025,
    "kmeans.dist_second@60": 3.1351140254811636,
    "kmeans.dist_second@119": 1.9857367773902603,
    "kmeans.weight_sum@25": 7.999488060097996,
    "kmeans.weight_sum@60": 12.473100285951407,
    "kmeans.weight_sum@119": 15.110060335371337,
    "kmeans.settled_frac@25": 0.5136725262938573,
    "kmeans.settled_frac@60": 0.8564127056253706,
    "kmeans.settled_frac@119": 0.9782071302132749,
    "kmeans.withheld_reason@25": None,
    "kmeans.withheld_reason@60": None,
    "kmeans.withheld_reason@119": None,
    "micro.cluster@25": 2,
    "micro.cluster@60": 7,
    "micro.cluster@119": 2,
    "micro.dist@25": 1.1517210881848035,
    "micro.dist@60": 2.535806113533875,
    "micro.dist@119": 0.6444533017493204,
    "micro.micro_id@25": 2,
    "micro.micro_id@60": 13,
    "micro.micro_id@119": 7,
    "micro.outlier@25": False,
    "micro.outlier@60": True,
    "micro.outlier@119": False,
    "micro.n_clusters@25": 2,
    "micro.n_clusters@60": 1,
    "micro.n_clusters@119": 1,
    "micro.n_micro@25": 4,
    "micro.n_micro@60": 3,
    "micro.n_micro@119": 4,
    "micro.weight_sum@25": 7.999488060097996,
    "micro.weight_sum@60": 12.473100285951407,
    "micro.weight_sum@119": 15.110060335371337,
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
    "ew_class.p_lo@119": 0.9706581254697835,
    "ew_class.p_hi@25": 0.9982307415352083,
    "ew_class.p_hi@60": 0.00019491576659006523,
    "ew_class.p_hi@119": 0.0293418745302166,
    "ew_class.weight_sum@25": 7.999488060097996,
    "ew_class.weight_sum@60": 12.473100285951407,
    "ew_class.weight_sum@119": 15.110060335371337,
    "ew_class.settled_frac@25": 0.5136725262938573,
    "ew_class.settled_frac@60": 0.8564127056253706,
    "ew_class.settled_frac@119": 0.9782071302132749,
    "ew_class.withheld_reason@25": None,
    "ew_class.withheld_reason@60": None,
    "ew_class.withheld_reason@119": None,
    "marginal.weight_sum@25": 7.999488060097996,
    "marginal.weight_sum@60": 12.473100285951407,
    "marginal.weight_sum@119": 15.110060335371337,
    "marginal.weight_sum[a/x0]@end": 14.643505957885951,
    "marginal.n_kish[a/x0]@end": 22.588085651395215,
    "marginal.corr[a/x0]@end": 0.38573773334037753,
    "marginal.beta[a/x0]@end": 1.2316780579628355,
    "marginal.t_stat[a/x0]@end": 1.89706698898827,
    "marginal.weight_sum[a/x1]@end": 14.643505957885951,
    "marginal.n_kish[a/x1]@end": 22.588085651395215,
    "marginal.corr[a/x1]@end": -0.8869994444789747,
    "marginal.beta[a/x1]@end": -0.7081876904377251,
    "marginal.t_stat[a/x1]@end": -8.715757847257636,
    "marginal.weight_sum[b/x0]@end": 14.25784941881905,
    "marginal.n_kish[b/x0]@end": 23.20892217040503,
    "marginal.corr[b/x0]@end": 0.41913584410223514,
    "marginal.beta[b/x0]@end": 0.9210410451370624,
    "marginal.t_stat[b/x0]@end": 2.1260076771757284,
    "marginal.weight_sum[b/x1]@end": 14.25784941881905,
    "marginal.n_kish[b/x1]@end": 23.20892217040503,
    "marginal.corr[b/x1]@end": -0.7634819469727878,
    "marginal.beta[b/x1]@end": -0.6068290466325642,
    "marginal.t_stat[b/x1]@end": -5.444279522504034,
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
    "corrchange.since_change@25": None,
    "corrchange.since_change@60": None,
    "corrchange.since_change@119": None,
    "corrchange.weight_sum@25": 11.0,
    "corrchange.weight_sum@60": 29.0,
    "corrchange.weight_sum@119": 57.0,
    "corrchange.settled_frac@25": None,
    "corrchange.settled_frac@60": None,
    "corrchange.settled_frac@119": None,
    "corrchange.withheld_reason@25": None,
    "corrchange.withheld_reason@60": None,
    "corrchange.withheld_reason@119": None,
    "corrchange.stat#0": 1.0511490148381897,
    "corrchange.stat#1": 0.9841580801406815,
    "corrchange.stat#2": 0.8994078028569636,
    "corrchange.stat#3": 0.7926014051597302,
    "hmm.filtered_0@25": 0.0019837560822976905,
    "hmm.filtered_0@60": 0.706545048537925,
    "hmm.filtered_0@119": 3.684331614556573e-07,
    "hmm.filtered_1@25": 0.9980162439177024,
    "hmm.filtered_1@60": 0.29345495146207495,
    "hmm.filtered_1@119": 0.9999996315668386,
    "hmm.predicted_0@25": 0.29240113431229137,
    "hmm.predicted_0@60": 0.5116753621005756,
    "hmm.predicted_0@119": 0.2119510684647354,
    "hmm.predicted_1@25": 0.7075988656877088,
    "hmm.predicted_1@60": 0.48832463789942426,
    "hmm.predicted_1@119": 0.7880489315352648,
    "hmm.state@25": 1,
    "hmm.state@60": 0,
    "hmm.state@119": 1,
    "hmm.loglik@25": -4.532665819206047,
    "hmm.loglik@60": -5.8850767257311,
    "hmm.loglik@119": -3.0867786940606114,
    "hmm.weight_sum@25": 7.999488060097996,
    "hmm.weight_sum@60": 12.473100285951407,
    "hmm.weight_sum@119": 15.110060335371337,
    "hmm.settled_frac@25": 0.5136725262938573,
    "hmm.settled_frac@60": 0.8564127056253706,
    "hmm.settled_frac@119": 0.9782071302132749,
    "hmm.withheld_reason@25": None,
    "hmm.withheld_reason@60": None,
    "hmm.withheld_reason@119": None,
    "rcov.weight_sum@25": 11.0,
    "rcov.weight_sum@60": 0.0,
    "rcov.weight_sum@119": 28.0,
    "deco.u@25": 0.4167923065360934,
    "deco.u@60": 0.9917268375727508,
    "deco.u@119": 0.8126243515712438,
    "deco.rho@25": 0.576990565121172,
    "deco.rho@60": -0.1046746141189773,
    "deco.rho@119": 0.1288237626639817,
    "deco.loglik@25": -3.9973745744499514,
    "deco.loglik@60": -6.086831238680252,
    "deco.loglik@119": -2.2061901908973724,
    "deco.weight_sum@25": 7.999488060097996,
    "deco.weight_sum@60": 12.473100285951407,
    "deco.weight_sum@119": 15.110060335371337,
    "deco.settled_frac@25": 0.5136725262938573,
    "deco.settled_frac@60": 0.8564127056253706,
    "deco.settled_frac@119": 0.9782071302132749,
    "deco.withheld_reason@25": None,
    "deco.withheld_reason@60": None,
    "deco.withheld_reason@119": None,
    "bocpd.p_change@25": 0.02628246584085018,
    "bocpd.p_change@60": 0.027730092663944658,
    "bocpd.p_change@119": 0.06922058473266716,
    "bocpd.run_mode@25": 11,
    "bocpd.run_mode@60": 29,
    "bocpd.run_mode@119": 49,
    "bocpd.run_mean@25": 10.530689528178458,
    "bocpd.run_mean@60": 23.803185886723877,
    "bocpd.run_mean@119": 45.47861212922035,
    "bocpd.pred_x0@25": 0.08622170400348289,
    "bocpd.pred_x0@60": 0.5193042747383524,
    "bocpd.pred_x0@119": -0.0756610549307658,
    "bocpd.pred_x1@25": 1.2136644279052489,
    "bocpd.pred_x1@60": 1.712608799264034,
    "bocpd.pred_x1@119": 2.3863758622295497,
    "bocpd.loglik@25": -5.554086833699265,
    "bocpd.loglik@60": -7.0867959219070595,
    "bocpd.loglik@119": -3.259762464576387,
    "bocpd.weight_sum@25": 11.0,
    "bocpd.weight_sum@60": 28.5,
    "bocpd.weight_sum@119": 57.0,
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
    "seqtest.weight_sum@25": 12.0,
    "seqtest.weight_sum@60": 30.0,
    "seqtest.weight_sum@119": 59.0,
    "seqtest.settled_frac@25": None,
    "seqtest.settled_frac@60": None,
    "seqtest.settled_frac@119": None,
    "seqtest.withheld_reason@25": None,
    "seqtest.withheld_reason@60": None,
    "seqtest.withheld_reason@119": None,
    "seqtest_compare.log_e_a_y0@25": 1.4759065198095778,
    "seqtest_compare.log_e_a_y0@60": 7.327588430413045,
    "seqtest_compare.log_e_a_y0@119": 2.822090994363675,
    "seqtest_compare.log_e_b_y0@25": 0.0,
    "seqtest_compare.log_e_b_y0@60": 0.0,
    "seqtest_compare.log_e_b_y0@119": 0.0,
    "seqtest_compare.wins_a_y0@25": 4,
    "seqtest_compare.wins_a_y0@60": 21,
    "seqtest_compare.wins_a_y0@119": 36,
    "seqtest_compare.wins_b_y0@25": 0,
    "seqtest_compare.wins_b_y0@60": 2,
    "seqtest_compare.wins_b_y0@119": 14,
    "seqtest_compare.weight_sum@25": 12.0,
    "seqtest_compare.weight_sum@60": 30.0,
    "seqtest_compare.weight_sum@119": 59.0,
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

#: Produced by `uv run python tests/test_golden_pipeline.py`, as `GOLDEN` is.
SWITCHED_GOLDEN: dict[str, float | str | None] = {
    "window.pred_y0@25": -4.68515442331697,
    "window.pred_y0@60": -0.2680628499022917,
    "window.pred_y0@119": -0.15784967395919602,
    "window.resid_y0@25": -0.16645140091774646,
    "window.resid_y0@60": -0.20397827891478626,
    "window.resid_y0@119": 0.10715090300096719,
    "window.weight_sum@25": 5.170963014289502,
    "window.weight_sum@60": 6.120750224253165,
    "window.weight_sum@119": 5.240558540773436,
    "window.settled_frac@25": 0.5136725262938573,
    "window.settled_frac@60": 0.8564127056253706,
    "window.settled_frac@119": 0.9782071302132749,
    "window.withheld_reason@25": None,
    "window.withheld_reason@60": None,
    "window.withheld_reason@119": None,
    "embargo.pred_y0@25": -0.297030662716732,
    "embargo.pred_y0@60": -0.6184279290422761,
    "embargo.pred_y0@119": -0.38922812978297516,
    "embargo.resid_y0@25": -4.554575161517985,
    "embargo.resid_y0@60": 0.14638680022519812,
    "embargo.resid_y0@119": 0.33852935882474633,
    "embargo.weight_sum@25": 5.255854811913398,
    "embargo.weight_sum@60": 10.867042954248578,
    "embargo.weight_sum@119": 12.473068731537921,
    "embargo.settled_frac@25": 0.5136725262938573,
    "embargo.settled_frac@60": 0.8564127056253706,
    "embargo.settled_frac@119": 0.9782071302132749,
    "embargo.withheld_reason@25": None,
    "embargo.withheld_reason@60": None,
    "embargo.withheld_reason@119": None,
    "datetime.pred_y0@25": -4.660751915259962,
    "datetime.pred_y0@60": -0.2697229622339732,
    "datetime.pred_y0@119": -0.1594190668274919,
    "datetime.resid_y0@25": -0.19085390897475385,
    "datetime.resid_y0@60": -0.20231816658310475,
    "datetime.resid_y0@119": 0.10872029586926307,
    "datetime.weight_sum@25": 6.4723416348901885,
    "datetime.weight_sum@60": 11.280849084162378,
    "datetime.weight_sum@119": 12.473068731537921,
    "datetime.settled_frac@25": 0.5136725262938573,
    "datetime.settled_frac@60": 0.8564127056253706,
    "datetime.settled_frac@119": 0.9782071302132749,
    "datetime.withheld_reason@25": None,
    "datetime.withheld_reason@60": None,
    "datetime.withheld_reason@119": None,
    "session_shrink.pred_y0@25": -4.660751915259962,
    "session_shrink.pred_y0@60": -0.2701333033839248,
    "session_shrink.pred_y0@119": -0.1597873898801008,
    "session_shrink.resid_y0@25": -0.19085390897475385,
    "session_shrink.resid_y0@60": -0.2019078254331531,
    "session_shrink.resid_y0@119": 0.10908861892187197,
    "session_shrink.weight_sum@25": 6.4723416348901885,
    "session_shrink.weight_sum@60": 11.280849084162378,
    "session_shrink.weight_sum@119": 12.429041490202788,
    "session_shrink.settled_frac@25": 0.5136725262938573,
    "session_shrink.settled_frac@60": 0.8564127056253706,
    "session_shrink.settled_frac@119": 0.9788030573836302,
    "session_shrink.withheld_reason@25": None,
    "session_shrink.withheld_reason@60": None,
    "session_shrink.withheld_reason@119": None,
    "pairwise.pred_y0@25": -5.22128388930027,
    "pairwise.pred_y0@60": -0.24393260276802842,
    "pairwise.pred_y0@119": -0.15722900030899223,
    "pairwise.resid_y0@25": 0.36967806506555423,
    "pairwise.resid_y0@60": -0.22810852604904952,
    "pairwise.resid_y0@119": 0.1065302293507634,
    "pairwise.weight_sum@25": 6.4723416348901885,
    "pairwise.weight_sum@60": 11.280849084162378,
    "pairwise.weight_sum@119": 12.473068731537921,
    "pairwise.settled_frac@25": 0.5136725262938573,
    "pairwise.settled_frac@60": 0.8564127056253706,
    "pairwise.settled_frac@119": 0.9782071302132749,
    "pairwise.withheld_reason@25": None,
    "pairwise.withheld_reason@60": None,
    "pairwise.withheld_reason@119": None,
    "origin.pred_y0@25": -4.451211194243234,
    "origin.pred_y0@60": -0.76168400788369,
    "origin.pred_y0@119": -0.3987697479966598,
    "origin.resid_y0@25": -0.40039462999148245,
    "origin.resid_y0@60": 0.289642879066612,
    "origin.resid_y0@119": 0.348070977038431,
    "origin.weight_sum@25": 6.4723416348901885,
    "origin.weight_sum@60": 11.280849084162378,
    "origin.weight_sum@119": 12.473068731537921,
    "origin.settled_frac@25": 0.5136725262938573,
    "origin.settled_frac@60": 0.8564127056253706,
    "origin.settled_frac@119": 0.9782071302132749,
    "origin.withheld_reason@25": None,
    "origin.withheld_reason@60": None,
    "origin.withheld_reason@119": None,
    "prior.pred_y0@25": -4.562228076997657,
    "prior.pred_y0@60": -0.2642888895255784,
    "prior.pred_y0@119": -0.1593967799747384,
    "prior.resid_y0@25": -0.2893777472370589,
    "prior.resid_y0@60": -0.20775223929149955,
    "prior.resid_y0@119": 0.10869800901650956,
    "prior.weight_sum@25": 6.4723416348901885,
    "prior.weight_sum@60": 11.280849084162378,
    "prior.weight_sum@119": 12.473068731537921,
    "prior.settled_frac@25": 0.5136725262938573,
    "prior.settled_frac@60": 0.8564127056253706,
    "prior.settled_frac@119": 0.9782071302132749,
    "prior.withheld_reason@25": None,
    "prior.withheld_reason@60": None,
    "prior.withheld_reason@119": None,
    "feature_sets.pred_y0__one@25": -1.6404570700536767,
    "feature_sets.pred_y0__one@60": -2.012733190571163,
    "feature_sets.pred_y0__one@119": -1.5914788198789864,
    "feature_sets.resid_y0__one@25": -3.2111487541810395,
    "feature_sets.resid_y0__one@60": 1.5406920617540854,
    "feature_sets.resid_y0__one@119": 1.5407800489207575,
    "feature_sets.pred_y0__both@25": -4.660751915259962,
    "feature_sets.pred_y0__both@60": -0.2697229622339732,
    "feature_sets.pred_y0__both@119": -0.1594190668274919,
    "feature_sets.resid_y0__both@25": -0.19085390897475385,
    "feature_sets.resid_y0__both@60": -0.20231816658310475,
    "feature_sets.resid_y0__both@119": 0.10872029586926307,
    "feature_sets.weight_sum@25": 6.4723416348901885,
    "feature_sets.weight_sum@60": 11.280849084162378,
    "feature_sets.weight_sum@119": 12.473068731537921,
    "feature_sets.settled_frac@25": 0.5136725262938573,
    "feature_sets.settled_frac@60": 0.8564127056253706,
    "feature_sets.settled_frac@119": 0.9782071302132749,
    "feature_sets.withheld_reason@25": None,
    "feature_sets.withheld_reason@60": None,
    "feature_sets.withheld_reason@119": None,
    "diagnostics.pred_y0__r0.000001@25": -4.660751915259962,
    "diagnostics.pred_y0__r0.000001@60": -0.2697229622339732,
    "diagnostics.pred_y0__r0.000001@119": -0.1594190668274919,
    "diagnostics.resid_y0__r0.000001@25": -0.19085390897475385,
    "diagnostics.resid_y0__r0.000001@60": -0.20231816658310475,
    "diagnostics.resid_y0__r0.000001@119": 0.10872029586926307,
    "diagnostics.pred_y0__r0.5@25": -3.7561264196136572,
    "diagnostics.pred_y0__r0.5@60": 0.31554261003977735,
    "diagnostics.pred_y0__r0.5@119": -0.10871340902440046,
    "diagnostics.resid_y0__r0.5@25": -1.095479404621059,
    "diagnostics.resid_y0__r0.5@60": -0.7875837388568553,
    "diagnostics.resid_y0__r0.5@119": 0.05801463806617163,
    "diagnostics.error_inflation_y0__r0.000001@25": 1.4843176752158107,
    "diagnostics.error_inflation_y0__r0.000001@60": 1.1812319980257089,
    "diagnostics.error_inflation_y0__r0.000001@119": 1.040873973086321,
    "diagnostics.error_inflation_y0__r0.5@25": 1.409364998814697,
    "diagnostics.error_inflation_y0__r0.5@60": 1.157007244603655,
    "diagnostics.error_inflation_y0__r0.5@119": 1.0393163626467838,
    "diagnostics.sigma_y0__r0.000001@25": 0.04722546616544854,
    "diagnostics.sigma_y0__r0.000001@60": 0.1372115392932475,
    "diagnostics.sigma_y0__r0.000001@119": 0.10036314090241635,
    "diagnostics.sigma_y0__r0.5@25": 0.2450256356138074,
    "diagnostics.sigma_y0__r0.5@60": 0.763881587323822,
    "diagnostics.sigma_y0__r0.5@119": 0.592950019155725,
    "diagnostics.zscore_y0__r0.000001@25": -4.041334569490981,
    "diagnostics.zscore_y0__r0.000001@60": -1.4744981918081381,
    "diagnostics.zscore_y0__r0.000001@119": 1.0832691652702702,
    "diagnostics.zscore_y0__r0.5@25": -4.470876697765937,
    "diagnostics.zscore_y0__r0.5@60": -1.0310285676816366,
    "diagnostics.zscore_y0__r0.5@119": 0.09784068840873988,
    "diagnostics.ic_y0__r0.000001@25": None,
    "diagnostics.ic_y0__r0.000001@60": 0.9978466202072177,
    "diagnostics.ic_y0__r0.000001@119": 0.9990654533620151,
    "diagnostics.ic_y0__r0.5@25": None,
    "diagnostics.ic_y0__r0.5@60": 0.9056179764815633,
    "diagnostics.ic_y0__r0.5@119": 0.9774853336299085,
    "diagnostics.r2_y0__r0.000001@25": None,
    "diagnostics.r2_y0__r0.000001@60": 0.9940917610083624,
    "diagnostics.r2_y0__r0.000001@119": 0.9980194631696558,
    "diagnostics.r2_y0__r0.5@25": None,
    "diagnostics.r2_y0__r0.5@60": 0.816882915409759,
    "diagnostics.r2_y0__r0.5@119": 0.9308693549353356,
    "diagnostics.hit_rate_y0__r0.000001@25": 1.0,
    "diagnostics.hit_rate_y0__r0.000001@60": 1.0,
    "diagnostics.hit_rate_y0__r0.000001@119": 1.0,
    "diagnostics.hit_rate_y0__r0.5@25": 1.0,
    "diagnostics.hit_rate_y0__r0.5@60": 0.8988632650572276,
    "diagnostics.hit_rate_y0__r0.5@119": 0.9354197371368904,
    "diagnostics.abs_resid_q0.5_y0__r0.000001@25": 0.047117875647668395,
    "diagnostics.abs_resid_q0.5_y0__r0.000001@60": 0.10400161384976526,
    "diagnostics.abs_resid_q0.5_y0__r0.000001@119": 0.08544642857142856,
    "diagnostics.abs_resid_q0.5_y0__r0.5@25": 0.24511329681274902,
    "diagnostics.abs_resid_q0.5_y0__r0.5@60": 0.6288819875776398,
    "diagnostics.abs_resid_q0.5_y0__r0.5@119": 0.39256840796019904,
    "diagnostics.abs_resid_q0.9_y0__r0.000001@25": 0.047117875647668395,
    "diagnostics.abs_resid_q0.9_y0__r0.000001@60": 0.1767524171270718,
    "diagnostics.abs_resid_q0.9_y0__r0.000001@119": 0.17284604519774013,
    "diagnostics.abs_resid_q0.9_y0__r0.5@25": 0.24511329681274902,
    "diagnostics.abs_resid_q0.9_y0__r0.5@60": 1.0077519379844961,
    "diagnostics.abs_resid_q0.9_y0__r0.5@119": 0.8710762331838565,
    "diagnostics.autocorr_y0__r0.000001@25": None,
    "diagnostics.autocorr_y0__r0.000001@60": -0.14548257378119903,
    "diagnostics.autocorr_y0__r0.000001@119": -0.21134234792237652,
    "diagnostics.autocorr_y0__r0.5@25": None,
    "diagnostics.autocorr_y0__r0.5@60": -0.27944465526984763,
    "diagnostics.autocorr_y0__r0.5@119": 0.0087558548905179,
    "diagnostics.drift_y0__r0.000001@25": False,
    "diagnostics.drift_y0__r0.000001@60": False,
    "diagnostics.drift_y0__r0.000001@119": False,
    "diagnostics.drift_y0__r0.5@25": False,
    "diagnostics.drift_y0__r0.5@60": False,
    "diagnostics.drift_y0__r0.5@119": False,
    "diagnostics.weight_sum@25": 6.4723416348901885,
    "diagnostics.weight_sum@60": 11.280849084162378,
    "diagnostics.weight_sum@119": 12.473068731537921,
    "diagnostics.settled_frac@25": 0.5136725262938573,
    "diagnostics.settled_frac@60": 0.8564127056253706,
    "diagnostics.settled_frac@119": 0.9782071302132749,
    "diagnostics.withheld_reason@25": None,
    "diagnostics.withheld_reason@60": None,
    "diagnostics.withheld_reason@119": None,
    "diagnostics.pred_y0__selected@25": -4.660751915259962,
    "diagnostics.pred_y0__selected@60": -0.2697229622339732,
    "diagnostics.pred_y0__selected@119": -0.1594190668274919,
    "diagnostics.selected_y0@25": "r0.000001",
    "diagnostics.selected_y0@60": "r0.000001",
    "diagnostics.selected_y0@119": "r0.000001",
    "diagnostics.pred_y0__averaged@25": -4.660751915254954,
    "diagnostics.pred_y0__averaged@60": -0.2697229622339181,
    "diagnostics.pred_y0__averaged@119": -0.15941906682749182,
    "diagnostics.scored_clock@25": 42.0,
    "diagnostics.scored_clock@60": 93.0,
    "diagnostics.scored_clock@119": 184.0,
    "diagnostics.learned_clock@25": 40.0,
    "diagnostics.learned_clock@60": 91.0,
    "diagnostics.learned_clock@119": 174.0,
    "formula.pred_fwd@25": 0.03271607966199878,
    "formula.pred_fwd@60": 0.13309799763766708,
    "formula.pred_fwd@119": 0.06743285763369189,
    "formula.resid_fwd@25": None,
    "formula.resid_fwd@60": None,
    "formula.resid_fwd@119": None,
    "formula.weight_sum@25": 5.027057497905452,
    "formula.weight_sum@60": 11.735722004229697,
    "formula.weight_sum@119": 12.655749650116945,
    "formula.settled_frac@25": 0.5136725262938573,
    "formula.settled_frac@60": 0.8564127056253706,
    "formula.settled_frac@119": 0.9782071302132749,
    "formula.withheld_reason@25": None,
    "formula.withheld_reason@60": None,
    "formula.withheld_reason@119": None,
    "hazard.p_change@25": 0.008004227824948607,
    "hazard.p_change@60": 0.004637941116834747,
    "hazard.p_change@119": 0.04507427186450985,
    "hazard.run_mode@25": 9,
    "hazard.run_mode@60": 26,
    "hazard.run_mode@119": 48,
    "hazard.run_mean@25": 7.585927118060897,
    "hazard.run_mean@60": 18.790493075215366,
    "hazard.run_mean@119": 31.87547987489265,
    "hazard.pred_x0@25": 0.008318682889687838,
    "hazard.pred_x0@60": 0.4620347445084317,
    "hazard.pred_x0@119": 0.16183016354957847,
    "hazard.pred_x1@25": 2.354434489065438,
    "hazard.pred_x1@60": 1.6133081165936614,
    "hazard.pred_x1@119": 2.3890559221915795,
    "hazard.loglik@25": -5.568907819829768,
    "hazard.loglik@60": -6.923161760351755,
    "hazard.loglik@119": -3.6947415246018056,
    "hazard.weight_sum@25": 9.0,
    "hazard.weight_sum@60": 26.0,
    "hazard.weight_sum@119": 48.0,
    "hazard.settled_frac@25": None,
    "hazard.settled_frac@60": None,
    "hazard.settled_frac@119": None,
    "hazard.withheld_reason@25": None,
    "hazard.withheld_reason@60": None,
    "hazard.withheld_reason@119": None,
    "prune.cluster@25": 1,
    "prune.cluster@60": 6,
    "prune.cluster@119": 6,
    "prune.dist@25": 1.4476125796712394,
    "prune.dist@60": 2.6659345467092606,
    "prune.dist@119": 0.2743875309651686,
    "prune.micro_id@25": 1,
    "prune.micro_id@60": 12,
    "prune.micro_id@119": 6,
    "prune.outlier@25": False,
    "prune.outlier@60": True,
    "prune.outlier@119": False,
    "prune.n_clusters@25": 2,
    "prune.n_clusters@60": 1,
    "prune.n_clusters@119": 1,
    "prune.n_micro@25": 3,
    "prune.n_micro@60": 3,
    "prune.n_micro@119": 2,
    "prune.weight_sum@25": 6.4723416348901885,
    "prune.weight_sum@60": 11.280849084162378,
    "prune.weight_sum@119": 12.473068731537921,
    "prune.settled_frac@25": 0.5136725262938573,
    "prune.settled_frac@60": 0.8564127056253706,
    "prune.settled_frac@119": 0.9782071302132749,
    "prune.withheld_reason@25": None,
    "prune.withheld_reason@60": None,
    "prune.withheld_reason@119": None,
}


def _same_numbers(got: dict[str, float | str | None], golden: dict[str, float | str | None]):
    assert set(got) == set(golden), (
        "the output schema changed: "
        f"added {sorted(set(got) - set(golden))}, removed {sorted(set(golden) - set(got))}"
    )
    for key, want in golden.items():
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


# An emptied table fails here as a changed schema: a skip would have turned
# the cross-platform check off without a word.
def test_the_pipeline_produces_the_same_numbers_everywhere():
    _same_numbers(signature(), GOLDEN)


def test_the_switches_produce_the_same_numbers_everywhere():
    """The second bank's numbers, compared as the first bank's are: the
    paths a platform is likeliest to move, which the first bank never ran
    (review 2026-10-06, TA12)."""
    _same_numbers(switched_signature(), SWITCHED_GOLDEN)


def test_the_stream_exercises_what_it_claims_to():
    """A guard on the fixture: if it stopped containing nulls or a clock gap,
    the golden comparison would still pass while covering less."""
    df = switched_stream()
    assert df["x1"].null_count() > 0, "no null feature"
    assert df["y0"].null_count() > 0, "no null target"
    assert df["label"].null_count() > 0, "no null label"
    assert df["label"].n_unique() == 3, "the label must hold both classes and null"
    assert df["g"].n_unique() == 2, "not two groups"
    assert df["session"].n_unique() == 2, "no session break"
    assert df["w"].n_unique() > 1, "weights are constant"
    gaps = df["t"].diff().drop_nulls()
    assert gaps.max() > 6.0, "no gap beyond gap_cap"
    assert gaps.min() > 0.0, "the clock must be strictly increasing"
    assert (df["wz"] == 0.0).sum() > 10, "no row of weight 0 for the second bank"
    assert df["ts"].dtype == pl.Datetime("us"), "no temporal clock for the second bank"


if __name__ == "__main__":
    # Regeneration, run as a script rather than a test that skips unless
    # asked: prints the table to paste over `GOLDEN` above.
    for name, sig in (("GOLDEN", signature()), ("SWITCHED_GOLDEN", switched_signature())):
        print(f"{name}: dict[str, float | str | None] = {{")
        for key, value in sig.items():
            print(f"    {key!r}: {value!r},")
        print("}")
