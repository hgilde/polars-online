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

    PRINT_GOLDEN=1 uv run pytest tests/test_golden_pipeline.py -s -k print
"""

import os

import numpy as np
import polars as pl
import pytest

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
    return pl.DataFrame(
        {
            "t": t,
            "x0": x0,
            "x1": x1,
            "y0": y,
            "w": w,
            "g": ["a" if i % 2 == 0 else "b" for i in range(n)],
            "session": ["m" if i < n // 2 else "n" for i in range(n)],
        }
    )


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
            **common,
        ),
        po.spec.rls("rls", ridge=1.0, **common),
        po.spec.kalman("kalman", coef_halflife=80.0, **common),
        po.spec.lasso("lasso", lasso_path=[0.2, 0.0], max_rows_between_solves=1, **common),
        po.spec.huber("huber", max_rows_between_solves=1, **common),
        po.spec.quantile("quantile", quantile=0.75, max_rows_between_solves=1, **common),
        po.spec.sgd("sgd", learning_rate=0.02, **common),
        po.spec.pa("pa", **common),
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
            sm_every=10,
            clock="t",
            max_dclock=6.0,
            halflife=25.0,
            weight="w",
            group="g",
            min_periods=4.0,
        ),
    ]


def signature() -> dict[str, float | None]:
    """Every non-coefficient output field, at three fixed rows."""
    out = po.ModelBank(specs()).fit_predict(stream())
    sig: dict[str, float | None] = {}
    for spec in specs():
        name = spec["name"]
        for field in po.spec.output_fields(spec):
            if field.startswith("coef"):
                continue  # a list, and a reporting cadence rather than a value
            values = out[name].struct.field(field).to_list()
            for row in PICKS:
                sig[f"{name}.{field}@{row}"] = values[row]
    return sig


#: Produced by `PRINT_GOLDEN=1 uv run pytest tests/test_golden_pipeline.py -s -k print`.
GOLDEN: dict[str, float | None] = {
    "ridge.pred_y0__r0.000001@25": -4.868442910414911,
    "ridge.pred_y0__r0.000001@60": -0.18087432320800145,
    "ridge.pred_y0__r0.000001@119": -0.16041676613206843,
    "ridge.resid_y0__r0.000001@25": 0.016837086180195193,
    "ridge.resid_y0__r0.000001@60": -0.2911668056090765,
    "ridge.resid_y0__r0.000001@119": 0.1097179951738396,
    "ridge.pred_y0__r0.5@25": -3.2050425489243026,
    "ridge.pred_y0__r0.5@60": -0.07180553816137292,
    "ridge.pred_y0__r0.5@119": -0.5269777959938127,
    "ridge.resid_y0__r0.5@25": -1.6465632753104136,
    "ridge.resid_y0__r0.5@60": -0.400235590655705,
    "ridge.resid_y0__r0.5@119": 0.47627902503558384,
    "ridge.sigma_y0__r0.000001@25": 0.38759694637415354,
    "ridge.sigma_y0__r0.000001@60": 0.2212348412123627,
    "ridge.sigma_y0__r0.000001@119": 0.13162890014791626,
    "ridge.sigma_y0__r0.5@25": 2.989993453787287,
    "ridge.sigma_y0__r0.5@60": 0.6475327742294054,
    "ridge.sigma_y0__r0.5@119": 0.8440595248098973,
    "ridge.resid_z_y0__r0.000001@25": 0.04343967706067035,
    "ridge.resid_z_y0__r0.000001@60": -1.316098332493598,
    "ridge.resid_z_y0__r0.000001@119": 0.8335403171381469,
    "ridge.resid_z_y0__r0.5@25": -0.550691264298518,
    "ridge.resid_z_y0__r0.5@60": -0.6180931785761798,
    "ridge.resid_z_y0__r0.5@119": 0.5642718446223961,
    "ridge.n_eff@25": 7.999488060097996,
    "ridge.n_eff@60": 12.473100285951407,
    "ridge.n_eff@119": 14.963784088176922,
    "rls.pred_y0@25": -4.706391171521188,
    "rls.pred_y0@60": -0.24224285328032535,
    "rls.pred_y0@119": -0.1602928667654287,
    "rls.resid_y0@25": -0.14521465271352785,
    "rls.resid_y0@60": -0.22979827553675258,
    "rls.resid_y0@119": 0.10959409580719986,
    "rls.n_eff@25": 7.999488060097996,
    "rls.n_eff@60": 12.473100285951407,
    "rls.n_eff@119": 14.963784088176922,
    "kalman.pred_y0@25": -2.7312829037141775,
    "kalman.pred_y0@60": -2.9615252045518297,
    "kalman.pred_y0@119": -0.21914171055962206,
    "kalman.resid_y0@25": -2.1203229205205387,
    "kalman.resid_y0@60": 2.489484075734752,
    "kalman.resid_y0@119": 0.16844293960139323,
    "kalman.n_eff@25": 7.999488060097996,
    "kalman.n_eff@60": 12.473100285951407,
    "kalman.n_eff@119": 14.963784088176922,
    "lasso.pred_y0__l0.2@25": -4.48197863620857,
    "lasso.pred_y0__l0.2@60": -0.10743087024739717,
    "lasso.pred_y0__l0.2@119": -0.26334262685327525,
    "lasso.resid_y0__l0.2@25": -0.36962718802614614,
    "lasso.resid_y0__l0.2@60": -0.36461025856968077,
    "lasso.resid_y0__l0.2@119": 0.21264385589504642,
    "lasso.pred_y0__l0@25": -4.868448205967198,
    "lasso.pred_y0__l0@60": -0.1808747495865375,
    "lasso.pred_y0__l0@119": -0.16041559516975357,
    "lasso.resid_y0__l0@25": 0.01684238173248165,
    "lasso.resid_y0__l0@60": -0.29116637923054045,
    "lasso.resid_y0__l0@119": 0.10971682421152473,
    "lasso.n_eff@25": 7.999488060097996,
    "lasso.n_eff@60": 12.473100285951407,
    "lasso.n_eff@119": 14.963784088176922,
    "lasso.lam_selected_y0@25": 0.0,
    "lasso.lam_selected_y0@60": 0.0,
    "lasso.lam_selected_y0@119": 0.0,
    "huber.pred_y0@25": -4.6843758303294525,
    "huber.pred_y0@60": -0.255954933162325,
    "huber.pred_y0@119": -0.16091537804221104,
    "huber.resid_y0@25": -0.16722999390526372,
    "huber.resid_y0@60": -0.21608619565475295,
    "huber.resid_y0@119": 0.1102166070839822,
    "huber.n_eff@25": 7.999488060097996,
    "huber.n_eff@60": 12.473100285951407,
    "huber.n_eff@119": 14.963784088176922,
    "quantile.pred_y0@25": -4.677469978382074,
    "quantile.pred_y0@60": -0.2253364176126762,
    "quantile.pred_y0@119": -0.09239345208361713,
    "quantile.resid_y0@25": -0.17413584585264186,
    "quantile.resid_y0@60": -0.24670471120440174,
    "quantile.resid_y0@119": 0.041694681125388294,
    "quantile.n_eff@25": 7.999488060097996,
    "quantile.n_eff@60": 12.473100285951407,
    "quantile.n_eff@119": 14.963784088176922,
    "sgd.pred_y0@25": -4.443842552576382,
    "sgd.pred_y0@60": -0.08184132740466754,
    "sgd.pred_y0@119": -0.13251990556028465,
    "sgd.resid_y0@25": -0.40776327165833415,
    "sgd.resid_y0@60": -0.3901998014124104,
    "sgd.resid_y0@119": 0.08182113460205581,
    "sgd.n_eff@25": 7.999488060097996,
    "sgd.n_eff@60": 12.473100285951407,
    "sgd.n_eff@119": 14.963784088176922,
    "pa.pred_y0@25": -4.090089774120203,
    "pa.pred_y0@60": -0.1320239199951636,
    "pa.pred_y0@119": -0.15725972713536898,
    "pa.resid_y0@25": -0.7615160501145128,
    "pa.resid_y0@60": -0.34001720882191433,
    "pa.resid_y0@119": 0.10656095617714015,
    "pa.n_eff@25": 7.999488060097996,
    "pa.n_eff@60": 12.473100285951407,
    "pa.n_eff@119": 14.963784088176922,
    "ftrl.pred_y0@25": -4.004533091781381,
    "ftrl.pred_y0@60": -0.2849015592079467,
    "ftrl.pred_y0@119": -0.19641552391486483,
    "ftrl.resid_y0@25": -0.8470727324533351,
    "ftrl.resid_y0@60": -0.18713956960913125,
    "ftrl.resid_y0@119": 0.145716752956636,
    "ftrl.n_eff@25": 7.999488060097996,
    "ftrl.n_eff@60": 12.473100285951407,
    "ftrl.n_eff@119": 14.963784088176922,
    "holt.pred_y0@25": -0.09807283291315116,
    "holt.pred_y0@60": 0.09878089276993715,
    "holt.pred_y0@119": -1.4381932751550712,
    "holt.resid_y0@25": -4.753532991321565,
    "holt.resid_y0@60": -0.5708220215870151,
    "holt.resid_y0@119": 1.3874945041968423,
    "holt.n_eff@25": 8.573837237596514,
    "holt.n_eff@60": 13.244185655943454,
    "holt.n_eff@119": 15.09397614533663,
    "moments.mean_x0@25": 0.14567573584298,
    "moments.mean_x0@60": 0.5319388857885934,
    "moments.mean_x0@119": 0.0790243604929474,
    "moments.mean_x1@25": 1.3170476393181616,
    "moments.mean_x1@60": 1.503050612553659,
    "moments.mean_x1@119": 2.044551436145137,
    "moments.var_x0@25": 1.1771092288363498,
    "moments.var_x0@60": 1.1318174907805318,
    "moments.var_x0@119": 1.0215819661245173,
    "moments.var_x1@25": 11.367059553916071,
    "moments.var_x1@60": 5.058876857932047,
    "moments.var_x1@119": 7.421903762699691,
    "moments.corr_x0_x1@25": 0.1950011675208187,
    "moments.corr_x0_x1@60": 0.31607532904190233,
    "moments.corr_x0_x1@119": 0.24826722240142626,
    "moments.n_eff@25": 7.999488060097996,
    "moments.n_eff@60": 12.473100285951407,
    "moments.n_eff@119": 14.963784088176922,
    "kmeans.cluster@25": 1,
    "kmeans.cluster@60": 0,
    "kmeans.cluster@119": 0,
    "kmeans.dist@25": 1.0657683037801138,
    "kmeans.dist@60": 2.0686144858219824,
    "kmeans.dist@119": 0.345516814446269,
    "kmeans.dist2@25": 2.9549386449346025,
    "kmeans.dist2@60": 3.1351140254811636,
    "kmeans.dist2@119": 1.8593223277713318,
    "kmeans.n_eff@25": 7.999488060097996,
    "kmeans.n_eff@60": 12.473100285951407,
    "kmeans.n_eff@119": 14.963784088176922,
}


@pytest.mark.skipif(not os.environ.get("PRINT_GOLDEN"), reason="regeneration only")
def test_print_golden():
    print("\nGOLDEN = {")
    for k, v in signature().items():
        print(f"    {k!r}: {v!r},")
    print("}")


@pytest.mark.skipif(not GOLDEN, reason="golden values not generated yet")
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
        assert abs(have - want) <= TOL * (1.0 + abs(want)), (
            f"{key}: {have!r} vs {want!r} (relative {abs(have - want) / (1 + abs(want)):.2e})"
        )


def test_the_stream_exercises_what_it_claims_to():
    """A guard on the fixture: if it stopped containing nulls or a clock gap,
    the golden comparison would still pass while covering less."""
    df = stream()
    assert df["x1"].null_count() > 0, "no null feature"
    assert df["y0"].null_count() > 0, "no null target"
    assert df["g"].n_unique() == 2, "not two groups"
    assert df["session"].n_unique() == 2, "no session break"
    assert df["w"].n_unique() > 1, "weights are constant"
    gaps = df["t"].diff().drop_nulls()
    assert gaps.max() > 6.0, "no gap beyond max_dclock"
    assert gaps.min() > 0.0, "the clock must be strictly increasing"
