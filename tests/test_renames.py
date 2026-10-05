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
    "level_halflife": "level_half_life",
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
                "model": {"type": "ew_ridge"},
                "targets": ["y"],
                "features": ["x"],
                "halflife": 10,
            },
            "halflife was renamed half_life",
        ),
        (
            {
                "name": "m",
                "model": {"type": "ew_ridge", "window": 5},
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


def test_rolling_metrics_takes_window_size():
    bank, out = _fit()
    df = out.with_columns(pl.int_range(pl.len()).cast(pl.Float64).alias("t"))
    by = po.eval.rolling_metrics(df, "m", clock="t", window_size=20.0, min_obs=5)
    assert "window_start" in by.columns and len(by) >= 2
    with pytest.raises(TypeError):
        po.eval.rolling_metrics(df, "m", clock="t", window=20.0, min_obs=5)  # type: ignore[call-arg]


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
