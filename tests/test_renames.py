"""Task 144: the public names follow Polars and say what they do, with no
alias (docs/PLAN.md, decided by the user on 2026-10-02: "Add all", and no
backward compatibility for outputs).

An old parameter is refused naming the new one, from a spec builder and from
a spec dict (the Rust side, which the TOML file shares); an old output name
is simply gone; the collisions the old grammar had (a target ``z_y`` beside
``y`` under ``emit_resid_z``, a feature named ``var`` beside ``pca``) are
gone with the renames; and what a spec can still render twice is refused by
the builder, naming the inputs. The API surface snapshot
(``test_api_surface.py``) holds the rest of the table.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import polars_online as po
from polars_online._spec import _RENAMED

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("release_probe", REPO / "scripts/release_probe.py")
assert _spec and _spec.loader
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

pytestmark = pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")

#: The parameter table of docs/PLAN.md task 144: old name -> new name.
RENAMED = {
    "halflife": "half_life",
    "long_halflife": "long_half_life",
    "coef_halflife": "coef_half_life",
    "revert_halflife": "revert_half_life",
    "select_halflife": "select_half_life",
    "level_halflife": "half_life",
    "trend_halflife": "trend_half_life",
    "label_delay": "embargo",
    "max_dclock": "gap_cap",
    "window": "window_size",
    "min_periods": "min_weight",
    "emit_resid_z": "emit_zscore",
    "scale_features": "standardize",
    "on_clock_reset": "restart_after_step_back",
    "min_backwards_jump": "restart_after_step_back",
    "ridge_decay": "ridge_scale",
    "add_intercept": "fit_intercept",
    "max_cd_iters": "max_iter",
    "cd_tol": "tol",
    "reset": "reset_on_flag",
    # Task 196 (docs/PLAN.md §18, N14 and N16): the row counts say so, and
    # holt's level takes the spec's `half_life`, which `level_halflife`
    # now names directly.
    "update_every": "update_every_rows",
    "split_merge_every": "split_merge_every_rows",
    "permute_every": "permute_every_rows",
    "level_half_life": "half_life",
}

#: The output names that are gone, as substrings no field may carry.
OLD_OUTPUTS = ("n_eff", "resid_z_", "lam_selected", "pcorr_", "absresid", "logscore")


def test_the_builders_table_is_the_plans():
    assert _RENAMED == RENAMED


@pytest.mark.parametrize("old", sorted(RENAMED))
def test_an_old_parameter_is_refused_naming_the_new_one(old):
    builder = po.spec.corrchange if old == "reset" else po.spec.ewridge
    kw = {"features": ["x0", "x1"]} if old == "reset" else {"targets": ["y"], "features": ["x"]}
    with pytest.raises(TypeError, match=f"{old} was renamed {RENAMED[old]}"):
        builder("m", **kw, **{old: 1.0})


@pytest.mark.parametrize(
    ("spec", "old"),
    [
        (
            {
                "name": "m",
                "model": {"type": "ewridge"},
                "targets": ["y"],
                "features": ["x"],
                "halflife": 10,
            },
            "halflife was renamed half_life",
        ),
        (
            {
                "name": "m",
                "model": {"type": "ewridge", "window": 5},
                "targets": ["y"],
                "features": ["x"],
            },
            "window was renamed window_size",
        ),
        (
            {"name": "m", "model": {"type": "corrchange", "reset": True}, "features": ["x0", "x1"]},
            "reset was renamed reset_on_flag",
        ),
    ],
)
def test_a_spec_dict_with_an_old_name_is_refused_naming_the_new_one(spec, old):
    """The Rust side, shared with the command line's TOML."""
    with pytest.raises(ValueError, match=old):
        po.ModelBank([spec])


def test_rls_ridge_is_refused_naming_delta(tmp_path, online_cli):
    """docs/PLAN.md task 195 (N11; review round 4, CA7): ``rls``'s prior
    strength is ``delta``, the classic RLS name for ``P₀ = I/δ``, and not
    ``ridge``, which in ``ewridge``, ``huber`` and ``quantile`` is a
    per-observation penalty that never fades. ``rls``'s is ``ewridge``'s
    ``ridge_scale="sum"`` prior. The old name is refused naming the new one,
    by the builder, a spec dict and a TOML file; ``ridge`` stays the other
    models' name, and a model with neither is told nothing about ``delta``."""
    from conftest import run_online

    kw = {"targets": ["y"], "features": ["x"], "half_life": 10.0}
    with pytest.raises(TypeError, match="ridge was renamed delta"):
        po.spec.rls("m", **kw, ridge=1.0)
    spec = po.spec.rls("m", **kw, delta=2.0)
    assert spec["model"]["delta"] == 2.0
    raw = dict(spec, model={"type": "rls", "ridge": 1.0})
    with pytest.raises(ValueError, match="ridge was renamed delta"):
        po.ModelBank([raw])
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    df.write_parquet(tmp_path / "in.parquet")
    res = run_online(
        online_cli,
        tmp_path,
        [raw],
        input=tmp_path / "in.parquet",
        output=tmp_path / "out.parquet",
        args=["--dry-run"],
        check=False,
    )
    assert res.returncode != 0 and "ridge was renamed delta" in res.stderr, res.stderr
    po.spec.ewridge("m", **kw, ridge=1.0)
    sgd = po.spec.sgd("m", **kw)
    with pytest.raises(ValueError) as refused:
        po.ModelBank([dict(sgd, model={**sgd["model"], "ridge": 1.0})])
    assert "unknown field `ridge`" in str(refused.value), refused.value
    assert "renamed" not in str(refused.value), refused.value


def test_no_output_carries_an_old_name():
    """Every spec of the release probe's workload, every kind the bank builds."""
    for name, builder, kw, _ in probe.WORKLOAD:
        fields = po.spec.output_fields(getattr(po.spec, builder)(name, **kw))
        bad = [f for f in fields if any(o in f for o in OLD_OUTPUTS)]
        assert not bad, (name, bad)
        if builder == "micro":
            assert not any(f == "micro" or f.startswith("micro@") for f in fields), fields


def _workload(name: str) -> dict:
    kw = next(kw for n, b, kw, _ in probe.WORKLOAD if n == name)
    builder = next(b for n, b, kw, _ in probe.WORKLOAD if n == name)
    return po.spec.output_fields(getattr(po.spec, builder)(name, **kw))


def test_the_new_output_names():
    diag = _workload("ridge_diag")
    assert "weight_sum" in diag and "zscore_y" in diag and "abs_resid_q0.1_y" in diag
    assert any(f.startswith("penalty_selected_") for f in _workload("lasso"))
    scores = _workload("ew_cov_scores")
    assert "partial_corr_x1_x2" in scores and "pc0_loading_x1" in scores
    assert "loglik" in _workload("bocpd")
    hmm = _workload("hmm") if any(n == "hmm" for n, *_ in probe.WORKLOAD) else None
    if hmm is None:
        pytest.fail("the workload has no hmm spec; the hmm names are untested")
    assert "filtered_0" in hmm and "predicted_0" in hmm and "p_0" not in hmm
    assert "micro_id" in _workload("micro")


def test_the_collisions_the_old_grammar_had_are_gone():
    """A target ``z_y`` beside ``y`` under ``emit_zscore`` (``resid_z_y`` was
    both the z-score of ``y`` and the residual of ``z_y``), and a feature
    named ``var``, ``share`` or ``score`` beside ``pca``."""
    s = po.spec.ewridge("m", targets=["y", "z_y"], features=["x"], half_life=10.0, emit_zscore=True)
    fields = po.spec.output_fields(s)
    assert fields.count("resid_z_y") == 1 and "zscore_y" in fields and "zscore_z_y" in fields
    c = po.spec.ew_cov(
        "c", features=["var", "share", "score"], stats=["mean"], pca=1, half_life=10.0
    )
    fields = po.spec.output_fields(c)
    assert len(fields) == len(set(fields))
    assert "pc0_var" in fields and "pc0_loading_var" in fields


def test_two_outputs_with_one_name_are_refused_by_the_builder_naming_the_inputs():
    """``corr_a_b_c`` is the correlation of ``a_b`` with ``c`` and of ``a``
    with ``b_c``: the one collision the columns' own names can still make."""
    with pytest.raises(ValueError, match="two outputs render to the same field name") as e:
        po.spec.ew_cov("c", features=["a_b", "c", "a", "b_c"], stats=["corr"], half_life=10.0)
    msg = str(e.value)
    assert (
        "corr_a_b_c" in msg
        and "['a_b', 'c']" in msg.replace('"', "'")
        and "['a', 'b_c']" in msg.replace('"', "'")
    )


def _fit():
    rng = np.random.default_rng(144)
    n = 60
    x = rng.standard_normal(n)
    df = pl.DataFrame({"x": x, "y": 2.0 * x + 0.1 * rng.standard_normal(n), "g": ["a"] * n})
    spec = po.spec.ewridge(
        "m", targets=["y"], features=["x"], half_life=10.0, min_weight=5.0, group="g"
    )
    bank = po.ModelBank([spec])
    return bank, bank.fit_predict(df)


def test_the_same_number_under_one_name_in_every_frame():
    bank, out = _fit()
    fields = out["m"].struct.fields
    assert "weight_sum" in fields and not any("n_eff" in f for f in fields)
    assert out["m"].struct.field("withheld_reason")[0] == "below_min_weight"
    for frame in (bank.coef(), bank.last_row(), bank.summary(), bank.closed_groups()):
        assert not any("n_eff" in c for c in frame.columns), frame.columns
    assert "weight_sum" in bank.coef().columns and "penalty" in bank.coef().columns
    assert "lambda" not in bank.coef().columns
    assert "weight_sum" in bank.last_row().columns
    assert "weight_sum" in bank.summary().columns
    assert "weight_sum" in bank.gram(0)[0] and "n_eff" not in bank.gram(0)[0]


def test_window_metrics_takes_every():
    """Task 144 renamed ``rolling_metrics``' ``window`` ``window_size``; task
    197 renamed the function ``window_metrics`` and the keyword ``every``."""
    bank, out = _fit()
    df = out.with_columns(pl.int_range(pl.len()).cast(pl.Float64).alias("t"))
    windows = po.eval.window_metrics(df, "m", clock="t", every=20.0, min_samples=5)
    assert "window_start" in windows.columns and len(windows) >= 2
    with pytest.raises(TypeError):
        po.eval.window_metrics(df, "m", clock="t", window=20.0)  # type: ignore[call-arg]


def test_restart_after_step_back_is_one_rule():
    """Unset, a step back is refused; given, a larger one restarts and a
    smaller one is a late row, refused."""
    df = pl.DataFrame({"t": [0.0, 10.0, 20.0, 15.0, 30.0], "x": [1.0] * 5, "y": [1.0] * 5})
    base = dict(targets=["y"], features=["x"], half_life=10.0, clock="t", gap_cap=50.0)
    with pytest.raises(ValueError, match="restart_after_step_back is unset.*set .* below it"):
        po.ModelBank([po.spec.ewridge("m", **base)]).fit_predict(df)
    with pytest.raises(ValueError, match="no more than restart_after_step_back = 5"):
        po.ModelBank([po.spec.ewridge("m", restart_after_step_back=5.0, **base)]).fit_predict(df)
    out = po.ModelBank([po.spec.ewridge("m", restart_after_step_back=4.0, **base)]).fit_predict(df)
    assert out["m"].struct.field("weight_sum")[3] == 0.0
    with pytest.raises(ValueError, match="restart_after_step_back needs clock"):
        po.spec.ewridge(
            "m", targets=["y"], features=["x"], half_life=10.0, restart_after_step_back=1.0
        )


def test_the_restart_edge_is_inclusive_at_a_millisecond_on_a_temporal_clock():
    """Task 159 (R2): a negative delta under a second was rounded at the
    scale of a second (``-1 ms`` read ``-1 + 0.999``, a last bit over), so on
    a Datetime clock a step back of exactly ``restart_after_step_back`` at 1
    to 3 ms restarted the model, where the rule makes it a late row; at 100
    ms, and on a number clock, it was refused. The edge is inclusive at
    every scale."""
    base = datetime(2024, 1, 2, 9, 30)
    for ms in (1, 3, 100):
        t = [base + timedelta(milliseconds=m) for m in (0, 10 * ms, 20 * ms, 19 * ms, 30 * ms)]
        df = pl.DataFrame({"t": t, "x": [1.0] * 5, "y": [1.0] * 5})
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x"],
            half_life="10s",
            clock="t",
            gap_cap="1h",
            restart_after_step_back=timedelta(milliseconds=ms),
        )
        with pytest.raises(ValueError, match="no more than restart_after_step_back"):
            po.ModelBank([spec]).fit_predict(df)


def test_with_windows_refuses_an_old_clock_name_in_like_and_in_a_keyword():
    """Review R1, F1: a hand-written ``like=`` dict with a clock key under its
    old name was read as not setting it, and a keyword under an old name was
    refused as a stray expression; both now name the new name."""
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0], "x": [1.0, 2.0, 3.0]})
    like = {
        "name": "m",
        "features": ["x"],
        "clock": "t",
        "gap_cap": 50.0,
        "on_clock_reset": "reset_state",
        "min_backwards_jump": 1.0,
    }
    with pytest.raises(TypeError, match="on_clock_reset was renamed restart_after_step_back"):
        po.stream.with_windows(df, f=po.ewm_mean("x", half_life=2.0), like=like)
    with pytest.raises(TypeError, match="max_dclock was renamed gap_cap"):
        po.stream.with_windows(df, f=po.ewm_mean("x", half_life=2.0), clock="t", max_dclock=5.0)


# --- task 196: the names in the specs, the models and the command line ------

#: The parameters task 196 renamed or dropped (docs/PLAN.md §18, N14 N16):
#: old name -> new name.
RENAMED_196 = {
    "update_every": "update_every_rows",
    "split_merge_every": "split_merge_every_rows",
    "permute_every": "permute_every_rows",
    "level_half_life": "half_life",
}


def test_the_task_196_renames_are_in_the_builders_table():
    for old, new in RENAMED_196.items():
        assert _RENAMED[old] == new, old
    assert _RENAMED["level_halflife"] == "half_life", "no chain through a refused name"


@pytest.mark.parametrize(
    ("builder", "kw", "old"),
    [
        (po.spec.kmeans, dict(features=["x0", "x1"], k=2), "update_every"),
        (po.spec.kmeans, dict(features=["x0", "x1"], k=2), "split_merge_every"),
        (po.spec.corrchange, dict(features=["x0", "x1"], kind="window"), "permute_every"),
        (po.spec.holt, dict(targets=["y"], half_life=None), "level_half_life"),
    ],
)
def test_a_task_196_parameter_is_refused_by_its_own_builder_naming_the_new_one(builder, kw, old):
    with pytest.raises(TypeError, match=f"{old} was renamed {RENAMED_196[old]}"):
        builder("m", **kw, **{old: 10})


@pytest.mark.parametrize(
    ("model", "old"),
    [
        (
            {"type": "kmeans", "k": 2, "update_every": 1},
            "update_every was renamed update_every_rows",
        ),
        (
            {"type": "kmeans", "k": 2, "split_merge_every": 10},
            "split_merge_every was renamed split_merge_every_rows",
        ),
        (
            {"type": "corrchange", "kind": "window", "span_rows": 20, "permute_every": 10},
            "permute_every was renamed permute_every_rows",
        ),
        ({"type": "holt", "level_half_life": 10.0}, "level_half_life was renamed half_life"),
        ({"type": "ew_ridge"}, "ew_ridge was renamed ewridge"),
    ],
)
def test_a_task_196_name_in_a_spec_dict_is_refused_naming_the_new_one(model, old):
    """The Rust side, shared with the command line's TOML: a model key, and
    the model's `type` itself (N1). Each dict is otherwise one the bank
    builds."""
    spec: dict = {"name": "m", "model": model, "features": ["x0", "x1"], "half_life": 10.0}
    if model["type"] in ("corrchange", "holt"):
        del spec["half_life"]
    if model["type"] == "holt":
        spec |= {"targets": ["y"], "features": []}
    if model["type"] == "ew_ridge":
        spec |= {"targets": ["y"], "features": ["x0"]}
    with pytest.raises(ValueError, match=old):
        po.ModelBank([spec])


def test_the_model_type_is_ewridge_everywhere():
    """N1: the builder's, the README's and the core's spelling is the tag."""
    s = po.spec.ewridge("m", targets=["y"], features=["x"], half_life=10.0)
    assert s["model"]["type"] == "ewridge"
    bank = po.ModelBank(
        [
            {
                "name": "m",
                "model": {"type": "ewridge"},
                "targets": ["y"],
                "features": ["x"],
                "half_life": 10.0,
            }
        ]
    )
    assert bank.specs[0]["model"]["type"] == "ewridge"


def test_lagcorr_is_refused_naming_lag_corr_and_lag_corr_is_written_everywhere():
    """N10: the `stats` value, the field prefix and the state say `lag_corr`,
    as `partial_corr` does, and `marginal`'s lists follow."""
    with pytest.raises(ValueError, match="lagcorr was renamed lag_corr"):
        po.spec.ew_cov("c", features=["x0", "x1"], stats=["lagcorr"], lags=[1], half_life=10.0)
    with pytest.raises(ValueError, match="lagcorr was renamed lag_corr"):
        po.ModelBank(
            [
                {
                    "name": "c",
                    "model": {"type": "ew_cov", "stats": ["lagcorr"], "lags": [1]},
                    "features": ["x0", "x1"],
                    "half_life": 10.0,
                }
            ]
        )
    s = po.spec.ew_cov("c", features=["x0", "x1"], stats=["lag_corr"], lags=[1], half_life=10.0)
    fields = po.spec.output_fields(s)
    assert "lag_corr_x0_x1_l1" in fields and not any("lagcorr" in f for f in fields)
    assert "lagcorr" not in po.ModelBank([s]).to_json()
    m = po.spec.marginal("m", targets=["y"], features=["x"], lags=[1], half_life=10.0)
    rng = np.random.default_rng(196)
    df = pl.DataFrame({"x": rng.standard_normal(50), "y": rng.standard_normal(50)})
    bank = po.ModelBank([m])
    bank.fit_predict(df)
    cols = bank.marginal("m").columns
    assert {"lag_corr_xx", "lag_corr_yy", "lag_corr_xy", "lag_corr_yx"} <= set(cols)
    assert not any("lagcorr" in c for c in cols)


def test_kmeans_writes_dist_second():
    """N7: the distance to the second-nearest centre, which `dist2` read as
    the square of `dist`."""
    s = po.spec.kmeans("k", features=["x0", "x1"], k=2, half_life=50.0)
    fields = po.spec.output_fields(s)
    assert "dist_second" in fields and "dist2" not in fields


def test_holt_takes_one_half_life():
    """N16: one knob, one dict shape, one TOML key."""
    s = po.spec.holt("m", targets=["y"], half_life=20.0)
    assert "level_half_life" not in s["model"] and s["half_life"] == 20.0


_FRAME = pl.DataFrame({"t": [0.0, 1.0, 2.0], "x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
_SPEC = po.spec.ewridge("m", targets=["y"], features=["x"], half_life=10.0)


def _chunk_calls():
    """Every call that takes the chunk size, as (name, call taking **kw)."""
    lf = _FRAME.lazy()
    fitted = po.ModelBank([_SPEC])
    fitted.fit_predict(_FRAME)
    window = po.ewm_mean("x", half_life=2.0)
    refresh = pl.DataFrame({"s": ["a", "b"], "t": [0.0, 1.0], "v": [1.0, 2.0]})
    refresh_kw = dict(series="s", names=["a", "b"], clock="t", value="v")
    return [
        ("ModelBank.fit", lambda **kw: po.ModelBank([_SPEC]).fit(lf, **kw)),
        (
            "ModelBank.fit_predict_batches",
            lambda **kw: list(po.ModelBank([_SPEC]).fit_predict_batches(lf, **kw)),
        ),
        ("lf.online.fit_predict", lambda **kw: lf.online.fit_predict([_SPEC], **kw).collect()),
        ("lf.online.predict", lambda **kw: lf.online.predict(fitted, **kw).collect()),
        ("po.fit_predict", lambda **kw: po.fit_predict(lf, [_SPEC], **kw).collect()),
        ("po.predict", lambda **kw: po.predict(lf, fitted, **kw).collect()),
        (
            "po.stream.with_windows",
            lambda **kw: po.stream.with_windows(lf, f=window, clock="t", gap_cap=5.0, **kw),
        ),
        (
            "lf.online.with_windows",
            lambda **kw: lf.online.with_windows(f=window, clock="t", gap_cap=5.0, **kw).collect(),
        ),
        (
            "df.online.with_windows",
            lambda **kw: _FRAME.online.with_windows(f=window, clock="t", gap_cap=5.0, **kw),
        ),
        (
            "po.stream.refresh_time",
            lambda **kw: po.stream.refresh_time(refresh.lazy(), **refresh_kw, **kw).collect(),
        ),
    ]


@pytest.mark.parametrize("which", range(10))
def test_chunk_rows_is_refused_naming_chunk_size_and_chunk_size_works(which):
    """N2: Polars' name on the call it feeds, `collect_batches(chunk_size=)`."""
    _, call = _chunk_calls()[which]
    with pytest.raises(TypeError, match="chunk_rows was renamed chunk_size"):
        call(chunk_rows=2)
    call(chunk_size=2)


# --- task 197: names in the helper modules (docs/PLAN.md §18, N3-N6, N24) ---


def _scored_fit() -> pl.DataFrame:
    """Two specs over two groups, with a clock: what every ``po.eval``
    function reads."""
    rng = np.random.default_rng(197)
    n = 200
    x = rng.standard_normal(n)
    df = pl.DataFrame(
        {
            "t": np.arange(n, dtype=float),
            "x": x,
            "y": 2.0 * x + 0.5 * rng.standard_normal(n),
            "g": ["a", "b"] * (n // 2),
        }
    )
    common = dict(targets=["y"], features=["x"], half_life=20.0, min_weight=5.0, group="g")
    return po.ModelBank(
        [po.spec.ewridge("m", **common), po.spec.ewridge("k", ridge=1.0, **common)]
    ).fit_predict(df)


#: Each ``po.eval`` function with a renamed keyword, called with the
#: keyword's old name: N3's ``by`` and ``min_obs``, and N4's ``window_size``.
_OLD_EVAL_KEYWORDS = {
    "metrics by": (lambda o: po.eval.metrics(o, "m", by=["g"]), "by", "group"),
    "metrics min_obs": (lambda o: po.eval.metrics(o, "m", min_obs=1), "min_obs", "min_samples"),
    "window_metrics by": (
        lambda o: po.eval.window_metrics(o, "m", clock="t", every=50.0, by=["g"]),
        "by",
        "group",
    ),
    "window_metrics min_obs": (
        lambda o: po.eval.window_metrics(o, "m", clock="t", every=50.0, min_obs=1),
        "min_obs",
        "min_samples",
    ),
    "window_metrics window_size": (
        lambda o: po.eval.window_metrics(o, "m", clock="t", window_size=50.0),
        "window_size",
        "every",
    ),
    "compare_specs by": (lambda o: po.eval.compare_specs(o, ["m"], by=["g"]), "by", "group"),
    "compare_specs min_obs": (
        lambda o: po.eval.compare_specs(o, ["m"], min_obs=1),
        "min_obs",
        "min_samples",
    ),
    "sums by": (lambda o: po.eval.sums(o, "m", by=["g"]), "by", "group"),
    "seqtest by": (lambda o: po.eval.seqtest(o, a="m", b="k", by=["g"]), "by", "group"),
    "from_sums min_obs": (
        lambda o: po.eval.from_sums(po.eval.sums(o, "m"), min_obs=1),
        "min_obs",
        "min_samples",
    ),
}


@pytest.mark.parametrize("case", sorted(_OLD_EVAL_KEYWORDS))
def test_an_eval_keyword_renamed_is_refused_naming_the_new_one(case):
    """N3 (review round 4, AP6): ``min_obs`` is Polars' ``min_samples`` for
    the same count, and ``by`` is ``group``, the specs' word since 0.12.0.
    N4 (YB5): ``window_metrics``' bucket width is ``every``, as Polars'
    ``group_by_dynamic(every=)``."""
    call, old, new = _OLD_EVAL_KEYWORDS[case]
    with pytest.raises(TypeError, match=f"^po\\.eval\\.[a-z_]+: {old} was renamed {new}$"):
        call(_scored_fit())


def test_the_new_eval_keywords_are_the_old_ones_renamed():
    out = _scored_fit()
    m = po.eval.metrics(out, "m", group=["g"], min_samples=150)
    assert m.is_empty()
    m = po.eval.metrics(out, "m", group=["g"], min_samples=50)
    scored = out.group_by("g").agg(pl.col("m").struct.field("pred_y").is_not_null().sum()).sort("g")
    assert m["g"].to_list() == ["a", "b"] and m["n"].to_list() == scored["pred_y"].to_list()
    s = po.eval.from_sums(po.eval.sums(out, "m", group=["g"]), min_samples=50)
    assert s["g"].to_list() == ["a", "b"]
    w = po.eval.window_metrics(out, "m", clock="t", every=50.0, group=["g"], min_samples=5)
    assert w["window_start"].unique().sort().to_list() == [0.0, 50.0, 100.0, 150.0]
    c = po.eval.compare_specs(out, ["m", "k"], group=["g"], min_samples=50)
    assert c["spec"].to_list() == ["m", "m", "k", "k"]


#: Each function taking ``group``, called with a bare string and with a list.
_GROUPED = {
    "metrics": lambda o, g: po.eval.metrics(o, "m", group=g, min_samples=1),
    "window_metrics": lambda o, g: po.eval.window_metrics(
        o, "m", clock="t", every=50.0, group=g, min_samples=1
    ),
    "compare_specs": lambda o, g: po.eval.compare_specs(o, ["m", "k"], group=g, min_samples=1),
    "sums": lambda o, g: po.eval.sums(o, "m", group=g),
    "seqtest": lambda o, g: po.eval.seqtest(o, a="m", b="k", group=g),
}


@pytest.mark.parametrize("fn", sorted(_GROUPED))
def test_group_takes_a_bare_string_as_one_key(fn):
    """YB16: ``by="group"`` iterated the string's characters and failed on a
    column ``g``. A bare string is one key, as in Polars' ``group_by`` and
    ``over``, and the specs' ``group``."""
    out = _scored_fit().rename({"g": "group"})
    call = _GROUPED[fn]
    one = call(out, "group")
    assert one.equals(call(out, ["group"]))
    if fn != "seqtest":
        assert one["group"].unique().sort().to_list() == ["a", "b"]


def test_rolling_metrics_is_window_metrics_with_every():
    """N4 (YB5): the buckets do not overlap, which in Polars is
    ``group_by_dynamic(every=)``; ``rolling`` is its overlapping window."""
    out = _scored_fit()
    with pytest.raises(
        TypeError, match="^po.eval.rolling_metrics was renamed po.eval.window_metrics"
    ):
        po.eval.rolling_metrics(out, "m", clock="t", window_size=50.0)
    assert "rolling_metrics" not in po.eval.__all__ and "window_metrics" in po.eval.__all__
    got = po.eval.window_metrics(out, "m", clock="t", every=50.0, min_samples=1)
    assert got["window_start"].to_list() == [0.0, 50.0, 100.0, 150.0]
    bucket = (pl.col("t") // 50.0).alias("bucket")
    scored = out.group_by(bucket).agg(pl.col("m").struct.field("pred_y").is_not_null().sum())
    assert got["n"].to_list() == scored.sort("bucket")["pred_y"].to_list()


def test_rows_seen_is_rows_fed():
    """Task 194 (N9, PA11): the count the frames call ``rows_fed``. The old
    method stays as a stub that raises naming the new one, as the helpers'
    renamed functions do (``tests/test_state_frames.py`` holds the count)."""
    bank = po.ModelBank([po.spec.ewridge("m", targets=["y"], features=["x"], half_life=10.0)])
    with pytest.raises(AttributeError, match=r"^ModelBank\.rows_seen was renamed rows_fed$"):
        bank.rows_seen()
    assert bank.rows_fed() == 0


def test_corr_shift_is_absorption_shift():
    """N5 (YB10): Polars' ``shift`` is a lag; this is Kritzman's
    standardised absorption shift."""
    fast, slow = np.array([0.5, 0.6, 0.7]), np.array([0.4, 0.4, 0.5])
    with pytest.raises(TypeError, match="^po.corr.shift was renamed po.corr.absorption_shift$"):
        po.corr.shift(fast, slow)
    assert "shift" not in po.corr.__all__ and "absorption_shift" in po.corr.__all__
    assert np.allclose(po.corr.absorption_shift(fast, slow, scale=0.1), [1.0, 2.0, 2.0])


def test_lasso_path_takes_penalties():
    """N6 (YB20): the spec takes the path as ``lasso_path=`` and ``coef()``
    reports each as ``penalty``; the offline path took them as ``lambdas``."""
    bank = po.ModelBank(
        [po.spec.ewridge("m", targets=["y"], features=["x0", "x1"], half_life=50.0)]
    )
    rng = np.random.default_rng(6)
    x = rng.standard_normal((100, 2))
    bank.fit_predict(pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": x @ [1.0, -0.5] + 0.1}))
    g = bank.gram("m")[0]
    want = po.gram.lasso_path(g, [0.1, 0.01])
    assert np.array_equal(po.gram.lasso_path(g, penalties=[0.1, 0.01]), want)
    with pytest.raises(TypeError, match="^po.gram.lasso_path: lambdas was renamed penalties$"):
        po.gram.lasso_path(g, lambdas=[0.1, 0.01])
    with pytest.raises(ValueError, match="penalties must be finite and >= 0"):
        po.gram.lasso_path(g, [-1.0])


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("gram", "INTERCEPT"),
        ("eval", "SUM_FIELDS"),
        ("eval", "RESERVED"),
        ("stream", "ROLE"),
        ("corr", "Z_CLIP"),
    ],
)
def test_a_module_constant_the_docs_name_is_exported(module, name):
    """N24 (YB9): docstrings link these with ``:data:``, tests and readers
    use them, and outside ``__all__`` the reference rendered none of them,
    so every ``:data:`` link was dead."""
    assert name in getattr(po, module).__all__
